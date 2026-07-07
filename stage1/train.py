# -*- coding: utf-8 -*-
import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torchvision.utils as tvu
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import ReflectionPairDataset
from model import build_model, load_weights


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return distributed, rank, local_rank, world_size


def is_main(rank: int) -> bool:
    return rank == 0


def cleanup(distributed: bool) -> None:
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def huber_like_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    c = 0.03
    diff = torch.sqrt((pred - gt) ** 2 + c**2)
    return (diff - c).mean()


def save_train_images(input_tensor, gt, pred, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tvu.save_image(input_tensor, out_dir / "input.png", normalize=True, value_range=(-1, 1))
    tvu.save_image(gt, out_dir / "gt.png")
    tvu.save_image(pred.clamp(0.0, 1.0), out_dir / "output.png")


def parse_args():
    parser = argparse.ArgumentParser("Stage1 reflection-only training")
    parser.add_argument("--data-root", default="dataset/RDRF_dataset", help="root containing train/LQ and train/LQ_ReflectionOnly")
    parser.add_argument("--output-dir", default="runs/stage1")
    parser.add_argument("--checkpoint", default=None, help="optional checkpoint to initialize or resume model weights")
    parser.add_argument("--unsafe-load", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--save-step", type=int, default=5000)
    parser.add_argument("--log-step", type=int, default=100)
    parser.add_argument("--vis-step", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--resume-step", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    distributed, rank, local_rank, world_size = init_distributed()

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    if is_main(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        print("--- Stage1 training ---")
        print(f"data_root: {args.data_root}")
        print(f"output_dir: {output_dir}")
        print(f"checkpoint: {args.checkpoint}")
        print(f"batch_size_per_rank: {args.batch_size}, world_size: {world_size}")

    model = build_model().to(device)
    if args.checkpoint:
        load_weights(model, args.checkpoint, device, args.unsafe_load)
        if is_main(rank):
            print("--- weight loaded ---")

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    dataset = ReflectionPairDataset(
        data_root=args.data_root,
        crop_size=tuple(args.crop_size),
        split="train",
        lq_dir_name="LQ",
        gt_dir_name="LQ_ReflectionOnly",
        random_flip=True,
        random_rotate=True,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    total_steps = args.resume_step
    epoch = 0
    model.train()
    try:
        while total_steps < args.num_steps:
            if sampler is not None:
                sampler.set_epoch(epoch)
            epoch += 1
            for input_tensor, gt, _ in loader:
                input_tensor = input_tensor.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                pred = model(input_tensor)
                loss = huber_like_loss(pred, gt)
                loss.backward()
                optimizer.step()

                total_steps += 1
                if is_main(rank) and total_steps % args.log_step == 0:
                    print(f"step={total_steps}, loss={float(loss.detach().cpu()):.6f}", flush=True)

                if is_main(rank) and total_steps % args.vis_step == 0:
                    save_train_images(input_tensor, gt, pred, output_dir / "train_res")

                if total_steps % args.save_step == 0:
                    if distributed:
                        dist.barrier()
                    if is_main(rank):
                        net = model.module if hasattr(model, "module") else model
                        torch.save(net.state_dict(), output_dir / f"{total_steps}_ckpt")
                        print(f"saved: {output_dir / f'{total_steps}_ckpt'}", flush=True)
                    if distributed:
                        dist.barrier()

                if total_steps >= args.num_steps:
                    if is_main(rank):
                        net = model.module if hasattr(model, "module") else model
                        torch.save(net.state_dict(), output_dir / "last_ckpt")
                        print(f"saved: {output_dir / 'last_ckpt'}")
                        print("Finish!")
                    return
    finally:
        cleanup(distributed)


if __name__ == "__main__":
    main()
