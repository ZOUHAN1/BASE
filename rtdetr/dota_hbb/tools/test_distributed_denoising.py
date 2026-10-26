"""Regression check for uneven empty-target batches across distributed ranks."""

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.zoo.rtdetr.denoising import get_contrastive_denoising_training_group


def make_targets(rank, device):
    if rank == 0:
        labels = torch.tensor([0, 2, 4], dtype=torch.int64, device=device)
        boxes = torch.tensor(
            [
                [0.25, 0.25, 0.10, 0.10],
                [0.50, 0.50, 0.20, 0.20],
                [0.75, 0.75, 0.15, 0.15],
            ],
            dtype=torch.float32,
            device=device,
        )
    else:
        labels = torch.empty(0, dtype=torch.int64, device=device)
        boxes = torch.empty((0, 4), dtype=torch.float32, device=device)
    return [{"labels": labels, "boxes": boxes}]


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    use_cuda = os.environ.get("TEST_DEVICE") == "cuda"
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend)
    class_embed = torch.nn.Embedding(16, 32, device=device)

    dn_class, dn_bbox, attn_mask, dn_meta = get_contrastive_denoising_training_group(
        make_targets(rank, device),
        num_classes=15,
        num_queries=300,
        class_embed=class_embed,
        num_denoising=100,
    )

    assert dn_class.shape == (1, 198, 32)
    assert dn_bbox.shape == (1, 198, 4)
    assert attn_mask.shape == (498, 498)
    assert dn_meta["dn_num_group"] == 33
    assert dn_meta["dn_num_split"] == [198, 300]
    expected_positive = 99 if rank == 0 else 0
    assert dn_meta["dn_positive_idx"][0].numel() == expected_positive

    if rank == 0:
        print("distributed denoising empty-target regression: PASS")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
