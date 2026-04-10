from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIGS = [
    "configs/ablations/spatial_only.yaml",
    "configs/ablations/fft_only.yaml",
    "configs/ablations/dct_only.yaml",
    "configs/ablations/fusion_no_attention.yaml",
    "configs/ablations/fusion_attention.yaml",
]


def run_step(module_name: str, config_path: str) -> None:
    command = [sys.executable, "-m", module_name, "--config", config_path]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full ablation suite.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="List of config files to run.",
    )
    args = parser.parse_args()

    for config_path in args.configs:
        print(f"\n=== Training: {config_path} ===")
        run_step("src.train_cnn", config_path)
        print(f"\n=== Evaluating: {config_path} ===")
        run_step("src.evaluate_cnn", config_path)

    summary_command = [
        sys.executable,
        "-m",
        "src.summarize_ablation",
        "--configs",
        *args.configs,
    ]
    subprocess.run(summary_command, check=True)


if __name__ == "__main__":
    main()
