import argparse
import os
import subprocess
import sys

import magnus
import torch

from ppo_train import Config


def parse_args() -> tuple[int, list[str]]:
    parser = argparse.ArgumentParser(description="Run PPO training and custody output")
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=0,
        help="Number of GPU processes. 0 means auto-detect.",
    )
    args, train_args = parser.parse_known_args()

    if args.nproc_per_node < 0:
        raise ValueError("--nproc-per-node must be >= 0")

    if args.nproc_per_node == 0:
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        nproc_per_node = max(1, gpu_count)
    else:
        nproc_per_node = args.nproc_per_node

    return nproc_per_node, train_args


def extract_ckpt_path(train_args: list[str]) -> str:
    ckpt_path = Config.ckpt_path
    for idx, token in enumerate(train_args):
        if token.startswith("--ckpt-path="):
            return token.split("=", 1)[1]
        if token == "--ckpt-path" and idx + 1 < len(train_args):
            return train_args[idx + 1]
    return ckpt_path


def build_train_command(nproc_per_node: int, train_args: list[str]) -> list[str]:
    if nproc_per_node > 1:
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={nproc_per_node}",
            "ppo_train.py",
            *train_args,
        ]
    return [sys.executable, "ppo_train.py", *train_args]


def main() -> None:
    nproc_per_node, train_args = parse_args()
    ckpt_path = extract_ckpt_path(train_args)
    cmd = build_train_command(nproc_per_node, train_args)

    print(f"start training with nproc_per_node={nproc_per_node}")
    subprocess.run(cmd, check=True)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"training output path not found: {ckpt_path}")

    file_secret = magnus.custody_file(ckpt_path)
    print(f"file secret: {file_secret}")


if __name__ == "__main__":
    main()
