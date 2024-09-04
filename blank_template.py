import time
import datetime
import warnings

import torch
import torch.nn as nn

from logger import Logger
from args import default_parser
import utils
import torchvision_references as ref

from models import get_model


def train_epoch() :
    print("TODO")

def evaluate() :
    print("TODO")

def main(args) :
    utils.init_distributed_mode(args)

    # Signal Handler to automatically relaunch slurm job
    utils.init_signal_handler(args)

    device = torch.device(args.device)

    # log only on main process
    if utils.is_main_process() :
        # similar API to wandb except mode and log_dir
        logger = Logger(project_name="whatever",
                run_name=args.name,
                tags=["patate"],
                resume=True,
                args=args,
                mode=args.logger,
                log_dir=args.output_dir)
        
    # OPTIONNALY : Use deterministic algorithms or not (cf example resnet)
    
    # Dataset Creation and Loading
    # Model Creation
    model = get_model(args)
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Parallelize the model using Distributed Data Parallel (DDP)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    # Loss, Criterion and optimizer (use model.parameters() for optimizer)
    # Load checkpoint
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch()
        lr_scheduler.step()
        evaluate(model, criterion, data_loader_test, device=device)
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
if __name__ == "__main" :
    args, unknown_args = default_parser().parse_known_args()
    main(args)