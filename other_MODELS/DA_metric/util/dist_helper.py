import os
import subprocess
import torch
import torch.distributed as dist

def setup_distributed(backend="nccl", port=None):
    """AdaHessian Optimizer
    Lifted from https://github.com/BIGBALLON/distribuuuu/blob/master/distribuuuu/utils.py
    Originally licensed MIT, Copyright (c) 2020 Wei Li
    """
    num_gpus = torch.cuda.device_count()

    if num_gpus > 1:
        # If SLURM is used, set environment variables accordingly
        if "SLURM_JOB_ID" in os.environ:
            rank = int(os.environ["SLURM_PROCID"])
            world_size = int(os.environ["SLURM_NTASKS"])
            node_list = os.environ["SLURM_NODELIST"]
            addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")

            os.environ["MASTER_PORT"] = str(port or 10685)
            os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", addr)
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["LOCAL_RANK"] = str(rank % num_gpus)
            os.environ["RANK"] = str(rank)
        else:
            # For local multi-GPU setup
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", num_gpus))

            os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
            os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    else:
        # For a single GPU setup
        rank = 0
        world_size = 1
        return rank, world_size

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return rank, world_size

import os
import subprocess
import torch
import torch.distributed as dist

def setup_distributed(backend="nccl", port=None):
    """AdaHessian Optimizer
    Lifted from https://github.com/BIGBALLON/distribuuuu/blob/master/distribuuuu/utils.py
    Originally licensed MIT, Copyright (c) 2020 Wei Li
    """
    num_gpus = torch.cuda.device_count()

    if num_gpus > 1:
        # If SLURM is used, set environment variables accordingly
        if "SLURM_JOB_ID" in os.environ:
            rank = int(os.environ["SLURM_PROCID"])
            #world_size = int(os.environ["SLURM_NTASKS"])
            world_size = num_gpus
            node_list = os.environ["SLURM_NODELIST"]
            addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")

            os.environ["MASTER_PORT"] = str(port or 10685)
            os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", addr)
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["LOCAL_RANK"] = str(rank % num_gpus)
            os.environ["RANK"] = str(rank)
        else:
            # For local multi-GPU setup
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", num_gpus))

            os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
            os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    else:
        # For a single GPU setup
        rank = 0
        world_size = 1
        return rank, world_size

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return rank, world_size

