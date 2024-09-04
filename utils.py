import os
import sys
import signal
import stat

import hostlist
import torch
import torch.distributed as dist

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        print("Using slurm")
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.slurm_jobid = int(os.environ["SLURM_JOB_ID"])
        args.slurm_jobname = os.environ["SLURM_JOB_NAME"]

        local_rank = int(os.environ['SLURM_LOCALID'])
        cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
        # get node list from slurm
        hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
        
        # get IDs of reserved GPU
        gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")
        print(gpu_ids)
        
        # define MASTER_ADD & MASTER_PORT
        os.environ['MASTER_ADDR'] = hostnames[0]
        os.environ['MASTER_PORT'] = str(10000 + args.slurm_jobid % 20000)
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True
    print(args.gpu)
    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url} {args.gpu} {args.world_size}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t

def create_dir(dir) :
    if is_main_process() :
        if not os.path.isdir(dir) :
            os.mkdir(dir)
            os.chmod(dir, stat.S_IRWXU | stat.S_IRWXO)

def sig_handler(signum, frame):
    prod_id = int(os.environ['SLURM_PROCID'])
    if prod_id == 0:
        os.system('scontrol requeue ' + os.environ['SLURM_JOB_ID'])
    sys.exit(-1)

def init_signal_handler():
    """
    Handle signals sent by SLURM for time limit.
    """
    signal.signal(signal.SIGUSR1, sig_handler)

def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path