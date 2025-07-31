import opentad.models.backbones.vit_adapter
import opentad.datasets.pku 
import sys
import opentad.datasets as datasets
import os

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler
from mmengine.config import Config, DictAction
from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores import train_one_epoch, val_one_epoch, eval_one_epoch, build_optimizer, build_scheduler
from opentad.utils import (
    set_seed,
    update_workdir,
    create_folder,
    save_config,
    setup_logger,
    ModelEma,
    save_checkpoint,
    save_best_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument("--resume", type=str, default=None, help="resume from a checkpoint")
    parser.add_argument("--not_eval", action="store_true", help="whether not to eval, only do inference")
    parser.add_argument("--disable_deterministic", action="store_true", help="disable deterministic for faster speed")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    parser.add_argument("--single_gpu", action="store_true", help="use single GPU training (non-distributed)")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Check if we should use distributed training
    use_distributed = not args.single_gpu and "LOCAL_RANK" in os.environ
    
    if use_distributed:
        # DDP init
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.rank = int(os.environ["RANK"])
        print(f"Distributed init (rank {args.rank}/{args.world_size}, local rank {args.local_rank})")
        
        # Disable libuv for Windows compatibility
        if os.name == 'nt':  # Windows
            os.environ.setdefault("USE_LIBUV", "0")
            
        dist.init_process_group("gloo", rank=args.rank, world_size=args.world_size)
        torch.cuda.set_device(args.local_rank)
    else:
        # Single GPU mode
        print("Using single GPU training")
        args.local_rank = 0
        args.world_size = 1
        args.rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    # set random seed, create work_dir, and save config
    set_seed(args.seed, args.disable_deterministic)
    cfg = update_workdir(cfg, args.id, args.world_size)
    if args.rank == 0:
        create_folder(cfg.work_dir)
        save_config(args.config, cfg.work_dir)

    # setup logger
    logger = setup_logger("Train", save_dir=cfg.work_dir, distributed_rank=args.rank)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    logger.info(f"Config: \n{cfg.pretty_text}")

    # build dataset
    train_dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=logger))
    
    if use_distributed:
        train_loader = build_dataloader(
            train_dataset,
            rank=args.rank,
            world_size=args.world_size,
            shuffle=True,
            drop_last=True,
            **cfg.solver.train,
        )
    else:
        # Single GPU dataloader
        train_loader = build_dataloader(
            train_dataset,
            rank=0,
            world_size=1,
            shuffle=True,
            drop_last=True,
            **cfg.solver.train,
        )

    # Validation dataset 추가
    val_dataset = None
    val_loader = None
    if hasattr(cfg.dataset, 'val'):
        val_dataset = build_dataset(cfg.dataset.val, default_args=dict(logger=logger))
        
        if use_distributed:
            val_loader = build_dataloader(
                val_dataset,
                rank=args.rank,
                world_size=args.world_size,
                shuffle=False,
                drop_last=False,
                **cfg.solver.val,
            )
        else:
            val_loader = build_dataloader(
                val_dataset,
                rank=0,
                world_size=1,
                shuffle=False,
                drop_last=False,
                **cfg.solver.val,
            )

    test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=logger))
    
    if use_distributed:
        test_loader = build_dataloader(
            test_dataset,
            rank=args.rank,
            world_size=args.world_size,
            shuffle=False,
            drop_last=False,
            **cfg.solver.test,
        )
    else:
        test_loader = build_dataloader(
            test_dataset,
            rank=0,
            world_size=1,
            shuffle=False,
            drop_last=False,
            **cfg.solver.test,
        )

    # build model
    model = build_detector(cfg.model)

    # DDP or single GPU
    if use_distributed:
        use_static_graph = getattr(cfg.solver, "static_graph", False)
        model = model.to(args.local_rank)
        model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False if use_static_graph else True,
            static_graph=use_static_graph,  # default is False, should be true when use activation checkpointing in E2E
        )
        logger.info(f"Using DDP with total {args.world_size} GPUS...")
    else:
        if torch.cuda.is_available():
            model = model.cuda()
        logger.info("Using single GPU training...")

    # FP16 compression (only for DDP)
    use_fp16_compress = getattr(cfg.solver, "fp16_compress", False)
    if use_fp16_compress and use_distributed:
        logger.info("Using FP16 compression ...")
        model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)

    # Model EMA
    use_ema = getattr(cfg.solver, "ema", False)
    if use_ema:
        logger.info("Using Model EMA...")
        model_ema = ModelEma(model)
    else:
        model_ema = None

    # AMP: automatic mixed precision
    use_amp = getattr(cfg.solver, "amp", False)
    if use_amp:
        logger.info("Using Automatic Mixed Precision...")
        scaler = GradScaler()
    else:
        scaler = None

    # PKU-MMD 학습을 위한 메모리 최적화
    if torch.cuda.is_available():
        # GPU 메모리 캐시 정리
        torch.cuda.empty_cache()
        # 메모리 할당 최적화
        torch.backends.cudnn.benchmark = True
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # build optimizer and scheduler
    optimizer = build_optimizer(cfg.optimizer, model, logger)
    scheduler, max_epoch = build_scheduler(cfg.scheduler, optimizer, len(train_loader))

    # override the max_epoch
    max_epoch = cfg.workflow.get("end_epoch", max_epoch)

    # resume: reset epoch, load checkpoint / best rmse
    if args.resume != None:
        logger.info("Resume training from: {}".format(args.resume))
        device = f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(args.resume, map_location=device)
        resume_epoch = checkpoint["epoch"]
        logger.info("Resume epoch is {}".format(resume_epoch))
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if model_ema != None:
            model_ema.module.load_state_dict(checkpoint["state_dict_ema"])

        del checkpoint  #  save memory if the model is very large such as ViT-g
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        resume_epoch = -1

    # train the detector
    logger.info("Training Starts...\n")
    for epoch in range(resume_epoch + 1, max_epoch):
        if use_distributed:
            train_loader.sampler.set_epoch(epoch)

        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            logger,
            model_ema=model_ema,
            clip_grad_l2norm=cfg.solver.clip_grad_norm,
            logging_interval=cfg.workflow.logging_interval,
            scaler=scaler,
        )


        if (val_loader is not None and 
            epoch >= cfg.workflow.val_start_epoch and
            (epoch + 1) % cfg.workflow.val_loss_interval == 0):
            
            val_one_epoch(
                val_loader,
                model,
                epoch,
                logger,
                model_ema=model_ema,
                scaler=scaler,
            )
    
        if (epoch == max_epoch - 1) or ((epoch + 1) % cfg.workflow.checkpoint_interval == 0):
            if args.rank == 0:
                save_checkpoint(model, model_ema, optimizer, scheduler, epoch, work_dir=cfg.work_dir)
                logger.info(f"Checkpoint saved at epoch {epoch}")

    logger.info("Training Over...\n")


if __name__ == "__main__":
    main()
