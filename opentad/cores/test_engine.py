import os
import copy
import json
import tqdm
import torch
import torch.distributed as dist

from opentad.utils import create_folder
from opentad.models.utils.post_processing import build_classifier, batched_nms
from opentad.evaluations import build_evaluator
from opentad.datasets.base import SlidingWindowDataset


def move_data_to_device(data_dict, device):
    """Move all tensors in data_dict to the specified device"""
    new_data_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            new_data_dict[key] = value.to(device)
        elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            new_data_dict[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
        else:
            new_data_dict[key] = value
    return new_data_dict


def eval_one_epoch(
    test_loader,
    model,
    cfg,
    logger,
    rank,
    model_ema=None,
    use_amp=False,
    world_size=0,
    not_eval=False,
):
    """Inference and Evaluation the model"""

    # load the ema dict for evaluation
    if model_ema != None:
        current_dict = copy.deepcopy(model.state_dict())
        model.load_state_dict(model_ema.module.state_dict())

    cfg.inference["folder"] = os.path.join(cfg.work_dir, "outputs")
    if cfg.inference.save_raw_prediction:
        create_folder(cfg.inference["folder"])

    # external classifier
    external_cls = None  # 기본값으로 초기화
    if "external_cls" in cfg.post_processing:
        if cfg.post_processing.external_cls != None:
            external_cls = build_classifier(cfg.post_processing.external_cls)
    else:
        external_cls = test_loader.dataset.class_map

    # whether the testing dataset is sliding window
    cfg.post_processing.sliding_window = isinstance(test_loader.dataset, SlidingWindowDataset)

    # model forward
    model.eval()
    result_dict = {}
    
    # Determine device
    device = next(model.parameters()).device
    
    for data_dict in tqdm.tqdm(test_loader, disable=(rank != 0)):
        # Move data to GPU
        data_dict = move_data_to_device(data_dict, device)
        
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                results = model(
                    **data_dict,
                    return_loss=False,
                    infer_cfg=cfg.inference,
                    post_cfg=cfg.post_processing,
                    ext_cls=external_cls,
                )

        # update the result dict
        for k, v in results.items():
            if k in result_dict.keys():
                result_dict[k].extend(v)
            else:
                result_dict[k] = v

    result_dict = gather_ddp_results(world_size, result_dict, cfg.post_processing)

    # load back the normal model dict
    if model_ema != None:
        model.load_state_dict(current_dict)

    if rank == 0:
        result_eval = dict(results=result_dict)
        if cfg.post_processing.save_dict:
            result_path = os.path.join(cfg.work_dir, "result_detection.json")
            with open(result_path, "w") as out:
                json.dump(result_eval, out)

        if not not_eval:
            # build evaluator
            evaluator = build_evaluator(dict(prediction_filename=result_eval, **cfg.evaluation))
            # evaluate and output
            logger.info("Evaluation starts...")
            metrics_dict = evaluator.evaluate()
            evaluator.logging(logger)


def gather_ddp_results(world_size, result_dict, post_cfg):
    # Check if distributed training is available and initialized
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            # Multi-GPU mode: gather results from all processes
            gather_dict_list = [None for _ in range(world_size)]
            dist.all_gather_object(gather_dict_list, result_dict)
            result_dict = {}
            for i in range(world_size):  # update the result dict
                for k, v in gather_dict_list[i].items():
                    if k in result_dict.keys():
                        result_dict[k].extend(v)
                    else:
                        result_dict[k] = v
        else:
            # Single-GPU mode: use result_dict as is
            pass
    except (ImportError, RuntimeError):
        # Single-GPU mode: use result_dict as is
        pass

    # do nms for sliding window, if needed
    if post_cfg.sliding_window == True and post_cfg.nms is not None:
        # assert sliding_window=True
        tmp_result_dict = {}
        for k, v in result_dict.items():
            segments = torch.Tensor([data["segment"] for data in v])
            scores = torch.Tensor([data["score"] for data in v])
            labels = []
            class_idx = []
            for data in v:
                if data["label"] not in class_idx:
                    class_idx.append(data["label"])
                labels.append(class_idx.index(data["label"]))
            labels = torch.Tensor(labels)

            segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)

            results_per_video = []
            for segment, label, score in zip(segments, labels, scores):
                # convert to python scalars
                results_per_video.append(
                    dict(
                        segment=[round(seg.item(), 2) for seg in segment],
                        label=class_idx[int(label.item())],
                        score=round(score.item(), 4),
                    )
                )
            tmp_result_dict[k] = results_per_video
        result_dict = tmp_result_dict
    return result_dict
