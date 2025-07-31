import opentad.models.backbones.vit_adapter
import opentad.datasets.pku  # pku_mmd 대신 pku로 수정
import os
import sys

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from mmengine.config import Config, DictAction
from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores import eval_one_epoch
from opentad.utils import update_workdir, set_seed, create_folder, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Test a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--checkpoint", type=str, default="none", help="the checkpoint path")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument("--not_eval", action="store_true", help="whether to not to eval, only do inference")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    parser.add_argument("--single_gpu", action="store_true", help="use single GPU testing (non-distributed)")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Check if we should use distributed testing
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
        print("Using single GPU testing")
        args.local_rank = 0
        args.world_size = 1
        args.rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    # set random seed, create work_dir
    set_seed(args.seed)
    if use_distributed:
        cfg = update_workdir(cfg, args.id, torch.cuda.device_count())
    else:
        cfg = update_workdir(cfg, args.id, 1)
    if args.rank == 0:
        create_folder(cfg.work_dir)

    # setup logger
    logger = setup_logger("Test", save_dir=cfg.work_dir, distributed_rank=args.rank)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    logger.info(f"Config: \n{cfg.pretty_text}")

    # build dataset
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
        model = model.to(args.local_rank)
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
        logger.info(f"Using DDP with total {args.world_size} GPUS...")
    else:
        if torch.cuda.is_available():
            model = model.cuda()
        logger.info("Using single GPU testing...")

    if cfg.inference.load_from_raw_predictions:  # if load with saved predictions, no need to load checkpoint
        logger.info(f"Loading from raw predictions: {cfg.inference.fuse_list}")
    else:  # load checkpoint: args -> config -> best
        if args.checkpoint != "none":
            checkpoint_path = args.checkpoint
        elif "test_epoch" in cfg.inference.keys():
            checkpoint_path = os.path.join(cfg.work_dir, f"checkpoint/epoch_{cfg.inference.test_epoch}.pth")
        else:
            checkpoint_path = os.path.join(cfg.work_dir, "checkpoint/best.pth")
        logger.info("Loading checkpoint from: {}".format(checkpoint_path))
        device = f"cuda:{args.rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        logger.info("Checkpoint is epoch {}.".format(checkpoint["epoch"]))

        # Model EMA
        use_ema = getattr(cfg.solver, "ema", False)
        if use_ema:
            model.load_state_dict(checkpoint["state_dict_ema"])
            logger.info("Using Model EMA...")
        else:
            model.load_state_dict(checkpoint["state_dict"])

    # AMP: automatic mixed precision
    use_amp = getattr(cfg.solver, "amp", False)
    if use_amp:
        logger.info("Using Automatic Mixed Precision...")

    # test the detector
    logger.info("Testing Starts...\n")
    eval_one_epoch(
        test_loader,
        model,
        cfg,
        logger,
        args.rank,
        model_ema=None,  # since we have loaded the ema model above
        use_amp=use_amp,
        world_size=args.world_size,
        not_eval=args.not_eval,
    )
    logger.info("Testing Over...\n")
    
    # PKU-MMD 평가 완료 후 추가 정보 출력
    if args.rank == 0 and not args.not_eval:
        logger.info("PKU-MMD evaluation completed successfully!")
        logger.info(f"Results saved in: {cfg.work_dir}")


if __name__ == "__main__":
    main()
