"""CLI entry point for Stage 1 RAE training.

Usage:
  # Single GPU
  python training/train_stage1_script.py --hdf5 /path/to/ph.hdf5

  # Multi-GPU on one machine (e.g., 2 GPUs)
  torchrun --nproc_per_node=2 training/train_stage1_script.py --hdf5 /path/to/ph.hdf5

  # Multi-node (e.g., 2 machines, 1 GPU each)
  # On node 0 (master):
  torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \\
    --master_addr=dh2020pc13.utm.utoronto.ca --master_port=29500 \\
    training/train_stage1_script.py --hdf5 /path/to/ph.hdf5
  # On node 1:
  torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \\
    --master_addr=dh2020pc13.utm.utoronto.ca --master_port=29500 \\
    training/train_stage1_script.py --hdf5 /path/to/ph.hdf5
"""

import argparse
import logging
import os
import sys

# Ensure repo root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.distributed as dist

from models.encoder import FrozenMultiViewEncoder
from models.adapter import TrainableAdapter
from models.decoder import ViTDecoder
from training.train_stage1 import Stage1Config, train_stage1


def main():
    parser = argparse.ArgumentParser(description="Stage 1 RAE Training")
    parser.add_argument("--hdf5", required=True, help="Path to unified HDF5 file")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--epoch_start_disc", type=int, default=6)
    parser.add_argument("--epoch_start_gan", type=int, default=8)
    parser.add_argument("--lr_gen", type=float, default=1e-4)
    parser.add_argument("--lr_disc", type=float, default=1e-4)
    parser.add_argument("--save_dir", default="checkpoints/stage1")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--pretrained_encoder", action="store_true", default=True)
    parser.add_argument("--no_pretrained_encoder", dest="pretrained_encoder", action="store_false")
    args = parser.parse_args()

    # Initialize distributed if launched with torchrun
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
    else:
        rank = 0

    # Logging only on rank 0
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s %(name)s %(message)s",
    )

    config = Stage1Config(
        hdf5_path=args.hdf5,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.num_epochs,
        epoch_start_disc=args.epoch_start_disc,
        epoch_start_gan=args.epoch_start_gan,
        lr_gen=args.lr_gen,
        lr_disc=args.lr_disc,
        save_dir=args.save_dir,
        save_every=args.save_every,
        disc_pretrained=True,
    )

    # Create models
    encoder = FrozenMultiViewEncoder(pretrained=args.pretrained_encoder)
    adapter = TrainableAdapter()
    decoder = ViTDecoder()

    train_stage1(
        config,
        encoder=encoder,
        adapter=adapter,
        decoder=decoder,
        resume_from=args.resume,
    )

    # Cleanup distributed
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
