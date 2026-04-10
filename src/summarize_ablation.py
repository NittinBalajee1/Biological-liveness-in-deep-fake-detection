from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .config import load_config
from .utils import ensure_dir


def describe_model(model_config: dict) -> str:
    parts = []
    if model_config.get("use_spatial", True):
        parts.append("Spatial")
    if model_config.get("use_fft", True):
        parts.append("FFT")
    if model_config.get("use_dct", True):
        parts.append("DCT")
    if model_config.get("use_attention_fusion", True) and len(parts) >= 2:
        parts.append("Attention")
    return " + ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ablation results.")
    parser.add_argument("--configs", nargs="+", required=True, help="Config files to summarize.")
    parser.add_argument(
        "--output-csv",
        default="outputs/ablations/ablation_summary.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/ablations/ablation_summary.md",
        help="Markdown output path.",
    )
    args = parser.parse_args()

    rows = []
    for config_path in args.configs:
        config = load_config(config_path)
        metrics_path = Path(config["output"]["metrics_path"])
        history_path = Path(config["output"]["history_path"])
        if not metrics_path.exists() or not history_path.exists():
            print(f"Skipping missing outputs for {config_path}")
            continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        history = json.loads(history_path.read_text(encoding="utf-8"))
        best_epoch = max(
            history.get("history", []),
            key=lambda row: row.get("val_metrics", {}).get("accuracy", -1.0),
            default=None,
        )
        row = {
            "config": config_path,
            "model": describe_model(config["model"]),
            "accuracy": metrics.get("accuracy", 0.0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "f1_score": metrics.get("f1_score", 0.0),
            "test_loss": metrics.get("test_loss", 0.0),
            "best_val_acc": (
                best_epoch.get("val_metrics", {}).get("accuracy", 0.0) if best_epoch else 0.0
            ),
            "best_val_f1": (
                best_epoch.get("val_metrics", {}).get("f1_score", 0.0) if best_epoch else 0.0
            ),
        }
        rows.append(row)

    rows.sort(key=lambda row: row["accuracy"], reverse=True)

    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    ensure_dir(output_csv.parent)
    ensure_dir(output_md.parent)

    fieldnames = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "test_loss",
        "best_val_acc",
        "best_val_f1",
        "config",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "# Ablation Summary",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | Test Loss | Best Val Acc | Best Val F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1_score:.4f} | {test_loss:.4f} | {best_val_acc:.4f} | {best_val_f1:.4f} |".format(
                **row
            )
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Ablation CSV: {output_csv}")
    print(f"Ablation Markdown: {output_md}")


if __name__ == "__main__":
    main()
