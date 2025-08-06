import copy
import torch
import tqdm
import time
from opentad.utils.misc import AverageMeter, reduce_loss


def get_model(model):
    """Get the actual model from either DDP wrapped model or single GPU model"""
    return model.module if hasattr(model, 'module') else model


def move_data_to_device(data_dict, device):
    """Move all data in data_dict to device"""
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.to(device)
        elif isinstance(value, list):
            data_dict[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
    return data_dict


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    logger,
    model_ema=None,
    clip_grad_l2norm=-1,
    logging_interval=50,  # 더 자주 로깅 (200 -> 50)
    scaler=None,
):
    """Training the model for one epoch"""

    num_iters = len(train_loader)
    logger.info("[Train]: Epoch {:d} started (Total iterations: {:d})".format(curr_epoch, num_iters))
    losses_tracker = {}
    use_amp = False if scaler is None else True
    
    # 에폭 시작 시간 기록
    epoch_start_time = time.time()
    iter_start_time = time.time()

    # Get the actual model (handle both DDP and single GPU cases)
    actual_model = get_model(model)

    # Determine device
    device = next(model.parameters()).device

    model.train()
    for iter_idx, data_dict in enumerate(train_loader):
        iter_data_start = time.time()
        
        optimizer.zero_grad()

        # Move data to GPU
        data_dict = move_data_to_device(data_dict, device)

        # current learning rate
        curr_backbone_lr = None
        if hasattr(actual_model, "backbone"):  # if backbone exists
            if actual_model.backbone.freeze_backbone == False:  # not frozen
                curr_backbone_lr = scheduler.get_last_lr()[0]
        curr_det_lr = scheduler.get_last_lr()[-1]

        # forward pass
        forward_start = time.time()
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            losses = model(**data_dict, return_loss=True)
        forward_time = time.time() - forward_start

        # compute the gradients
        backward_start = time.time()
        if use_amp:
            scaler.scale(losses["cost"]).backward()
        else:
            losses["cost"].backward()
        backward_time = time.time() - backward_start

        # gradient clipping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)

        # update parameters
        optimizer_start = time.time()
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer_time = time.time() - optimizer_start

        # update scheduler
        scheduler.step()

        # update ema
        if model_ema is not None:
            model_ema.update(model)

        # track all losses
        losses = reduce_loss(losses)  # only for log
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

        # printing each logging_interval
        if ((iter_idx != 0) and (iter_idx % logging_interval) == 0) or ((iter_idx + 1) == num_iters):
            # 시간 계산
            elapsed_time = time.time() - epoch_start_time
            iter_elapsed = time.time() - iter_start_time
            avg_time_per_iter = elapsed_time / (iter_idx + 1)
            remaining_iters = num_iters - (iter_idx + 1)
            estimated_remaining_time = remaining_iters * avg_time_per_iter
            
            # 진행률 계산
            progress_percent = (iter_idx + 1) / num_iters * 100
            
            # GPU 메모리 정보
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated() / 1024.0 / 1024.0
                gpu_memory_reserved = torch.cuda.memory_reserved() / 1024.0 / 1024.0
                gpu_memory_max = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            else:
                gpu_memory_allocated = gpu_memory_reserved = gpu_memory_max = 0
            
            # 로깅 메시지 구성
            block1 = "[Train]: [{:03d}][{:05d}/{:05d}] ({:.1f}%)".format(curr_epoch, iter_idx, num_iters - 1, progress_percent)
            block2 = "Loss={:.4f}".format(losses_tracker["cost"].avg)
            block3 = ["{:s}={:.4f}".format(key, value.avg) for key, value in losses_tracker.items() if key != "cost"]
            block4 = "lr_det={:.1e}".format(curr_det_lr)
            if curr_backbone_lr is not None:
                block4 = "lr_backbone={:.1e}".format(curr_backbone_lr) + "  " + block4
            
            # GPU 메모리 정보 추가
            block5 = "GPU={:.0f}MB(alloc)/{:.0f}MB(reserved)/{:.0f}MB(max)".format(
                gpu_memory_allocated, gpu_memory_reserved, gpu_memory_max
            )
            
            # 시간 정보 추가
            block6 = "ETA={:.0f}s".format(estimated_remaining_time)
            block7 = "iter_time={:.3f}s".format(iter_elapsed)
            
            # 상세 시간 분석 (첫 번째 로깅에서만)
            if iter_idx == logging_interval:
                block8 = "fwd={:.3f}s/bwd={:.3f}s/opt={:.3f}s".format(forward_time, backward_time, optimizer_time)
                logger.info("  ".join([block1, block2, "  ".join(block3), block4, block5, block6, block7, block8]))
            else:
                logger.info("  ".join([block1, block2, "  ".join(block3), block4, block5, block6, block7]))
            
            # 다음 로깅을 위한 시간 초기화
            iter_start_time = time.time()

    # 에폭 완료 요약
    epoch_total_time = time.time() - epoch_start_time
    avg_time_per_iter = epoch_total_time / num_iters
    
    logger.info(f"[Train]: Epoch {curr_epoch} completed in {epoch_total_time:.1f}s (avg {avg_time_per_iter:.3f}s/iter)")
    logger.info(f"[Train]: Final Loss={losses_tracker['cost'].avg:.4f}")


def val_one_epoch(
    val_loader,
    model,
    logger,
    rank,
    curr_epoch,
    model_ema=None,
    use_amp=False,
    evaluation=None,  # mAP 평가를 위한 파라미터 추가
):
    """Validating the model for one epoch: compute the loss and mAP"""

    # load the ema dict for evaluation
    if model_ema != None:
        current_dict = copy.deepcopy(model.state_dict())
        model.load_state_dict(model_ema.module.state_dict())

    logger.info("[Val]: Epoch {:d} Loss".format(curr_epoch))
    losses_tracker = {}
    result_dict = {}  # 예측 결과 수집을 위한 딕셔너리

    # Determine device
    device = next(model.parameters()).device

    model.eval()
    for data_dict in tqdm.tqdm(val_loader, disable=(rank != 0)):
        # Move data to GPU
        data_dict = move_data_to_device(data_dict, device)
        
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                # 손실 계산
                losses = model(**data_dict, return_loss=True)
                
                # 예측 수행 (mAP 평가를 위해)
                if evaluation is not None:
                    results = model(
                        **data_dict,
                        return_loss=False,
                        infer_cfg=dict(
                            load_from_raw_predictions=False,
                            save_raw_prediction=False,
                            folder="",
                        ),
                        post_cfg=dict(save_dict=False, sliding_window=True, nms=None),
                        ext_cls=val_loader.dataset.class_map,
                    )
                    
                    # 예측 결과 수집
                    for k, v in results.items():
                        if k in result_dict.keys():
                            result_dict[k].extend(v)
                        else:
                            result_dict[k] = v

        # track all losses
        losses = reduce_loss(losses)  # only for log
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

    # print to terminal
    block1 = "[Val]: [{:03d}]".format(curr_epoch)
    block2 = "Loss={:.4f}".format(losses_tracker["cost"].avg)
    block3 = ["{:s}={:.4f}".format(key, value.avg) for key, value in losses_tracker.items() if key != "cost"]
    
    # mAP 평가 추가
    mAP_results = ""
    if evaluation is not None and len(result_dict) > 0:
        try:
            # 예측 결과를 evaluation 형식으로 변환
            result_eval = dict(results=result_dict)
            
            # mAP 평가 실행
            from opentad.evaluations import build_evaluator
            # evaluation 설정을 복사하고 prediction_filename을 덮어쓰기
            eval_config = evaluation.copy()
            eval_config['prediction_filename'] = result_eval
            evaluator = build_evaluator(eval_config)
            mAP_metrics = evaluator.evaluate()
            
            # mAP 결과를 문자열로 변환
            mAP_parts = []
            for tiou, mAP in zip(evaluation.get('tiou_thresholds', []), mAP_metrics.get('mAPs', [])):
                mAP_parts.append(f"mAP@{tiou}={mAP*100:.2f}%")
            mAP_parts.append(f"Average-mAP={mAP_metrics.get('average_mAP', 0)*100:.2f}%")
            mAP_results = "  " + "  ".join(mAP_parts)
            
        except Exception as e:
            logger.warning(f"mAP evaluation failed: {e}")
            mAP_results = ""
    
    logger.info("  ".join([block1, block2, "  ".join(block3)]) + mAP_results)

    # load back the normal model dict
    if model_ema != None:
        model.load_state_dict(current_dict)
    return losses_tracker["cost"].avg
