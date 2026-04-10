"""
Comprehensive plotting script for training metrics.

Usage:
    # Compare specific runs:
    python training/plot_metrics.py \
        checkpoints/v3_lift_d256_0405_0405_1727 \
        checkpoints/v3_lift_d512_0405_0405_1707 \
        --labels "d=256" "d=512" \
        --title "Lift: d=256 vs d=512"

    # Multi-seed with confidence intervals:
    python training/plot_metrics.py \
        checkpoints/v3_lift_d512_seed42 \
        checkpoints/v3_lift_d512_seed43 \
        checkpoints/v3_lift_d512_seed44 \
        --group-by-config \
        --title "Lift d=512 (multi-seed)"

    # Auto-discover runs matching a pattern:
    python training/plot_metrics.py --glob "v3_lift_d512_*" --group-by-config
"""

import argparse
import glob as globmod
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


def load_metrics(run_dir):
    """Load metrics.jsonl from a run directory."""
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        print(f"Warning: {path} not found, skipping", file=sys.stderr)
        return None, None

    config = None
    epochs = []
    eval_rates = {}
    train_loss = {}
    val_loss = {}

    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("type") == "run_info":
                config = d.get("config", {})
                continue
            epoch = d.get("epoch")
            if epoch is None:
                continue
            if "eval_success_rate" in d:
                eval_rates[epoch] = d["eval_success_rate"]
            if "train" in d:
                train_loss[epoch] = d["train"].get("policy", d["train"].get("total"))
            if "valid" in d:
                val_loss[epoch] = d["valid"].get("policy")

    data = {
        "eval": eval_rates,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    return config, data


def config_key(config):
    """Extract a hashable key from config for grouping seeds."""
    if config is None:
        return "unknown"
    return (
        config.get("eval_task", "?"),
        config.get("d_model", "?"),
        config.get("spatial_pool_size", "?"),
        config.get("n_cond_layers", "?"),
        config.get("use_flow_matching", False),
        config.get("stage1_checkpoint", "") != "",
    )


def config_label(config):
    """Generate a human-readable label from config."""
    if config is None:
        return "unknown"
    task = config.get("eval_task", "?")
    d = config.get("d_model", "?")
    fm = "fm" if config.get("use_flow_matching") else "ddpm"
    ws = "ws" if config.get("stage1_checkpoint", "") else "no-ws"
    return f"{task} d={d} {fm} {ws}"


def plot_comparison(runs, labels, title, output, max_epoch=None, smoothing=0):
    """Plot success rate and loss curves for multiple runs."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    colors = plt.cm.tab10.colors
    plot_data = {"eval": axes[0], "train_loss": axes[1], "val_loss": axes[2]}
    titles = {"eval": "Eval Success Rate", "train_loss": "Train Loss", "val_loss": "Val Loss"}

    for i, (data, label) in enumerate(zip(runs, labels)):
        color = colors[i % len(colors)]
        for key, ax in plot_data.items():
            series = data.get(key, {})
            if not series:
                continue
            epochs = sorted(series.keys())
            if max_epoch is not None:
                epochs = [e for e in epochs if e <= max_epoch]
            values = [series[e] for e in epochs]

            if smoothing > 0 and len(values) > smoothing:
                kernel = np.ones(smoothing) / smoothing
                smoothed = np.convolve(values, kernel, mode="valid")
                smooth_epochs = epochs[smoothing - 1 :]
                ax.plot(smooth_epochs, smoothed, color=color, label=label, linewidth=2)
                ax.plot(epochs, values, color=color, alpha=0.2, linewidth=0.8)
            else:
                ax.plot(epochs, values, color=color, label=label, linewidth=1.5)

    for key, ax in plot_data.items():
        ax.set_title(titles[key])
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if key == "eval":
            ax.set_ylabel("Success Rate")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.set_ylim(-0.05, 1.05)
        else:
            ax.set_ylabel("Loss")

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


def plot_multiseed(grouped, title, output, max_epoch=None, cummax=False):
    """Plot with confidence intervals from multiple seeds per config."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    colors = plt.cm.tab10.colors
    metrics = [("eval", "Eval Success Rate"), ("train_loss", "Train Loss"), ("val_loss", "Val Loss")]

    for i, (group_label, data_list) in enumerate(grouped.items()):
        color = colors[i % len(colors)]
        n_seeds = len(data_list)

        for (key, metric_title), ax in zip(metrics, axes):
            all_epochs = set()
            for data in data_list:
                all_epochs.update(data.get(key, {}).keys())
            if not all_epochs:
                continue

            epochs = sorted(all_epochs)
            if max_epoch is not None:
                epochs = [e for e in epochs if e <= max_epoch]

            matrix = []
            for data in data_list:
                series = data.get(key, {})
                row = [series.get(e, np.nan) for e in epochs]
                matrix.append(row)
            matrix = np.array(matrix)

            mean = np.nanmean(matrix, axis=0)
            std = np.nanstd(matrix, axis=0)
            n_valid = np.sum(~np.isnan(matrix), axis=0)
            se = std / np.sqrt(np.maximum(n_valid, 1))

            if cummax and key == "eval":
                # Max across seeds with SD band below
                center = np.nanmax(matrix, axis=0)
                label = f"{group_label} (n={n_seeds}, max)"
                ax.plot(epochs, center, color=color, label=label, linewidth=2.5, marker="o", markersize=4)
                ax.fill_between(epochs, center - std, center + std, color=color, alpha=0.2)
            else:
                # Mean with prominent SD band
                label = f"{group_label} mean (n={n_seeds})"
                ax.plot(epochs, mean, color=color, label=label, linewidth=2.5, marker="o", markersize=3)
                ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.25)

    for (key, metric_title), ax in zip(metrics, axes):
        ax.set_title(metric_title)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if key == "eval":
            ax.set_ylabel("Success Rate")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.set_ylim(-0.05, 1.05)
        else:
            ax.set_ylabel("Loss")

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument("runs", nargs="*", help="Run directories to plot")
    parser.add_argument("--glob", type=str, help="Glob pattern for run dirs (relative to checkpoints/)")
    parser.add_argument("--labels", nargs="*", help="Labels for each run (same order as runs)")
    parser.add_argument("--title", type=str, default="Training Comparison")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file (default: auto)")
    parser.add_argument("--max-epoch", type=int, default=None)
    parser.add_argument("--smoothing", type=int, default=0, help="Moving average window for smoothing")
    parser.add_argument(
        "--group-by-config",
        action="store_true",
        help="Group runs by config (for multi-seed CI plots)",
    )
    parser.add_argument(
        "--cummax",
        action="store_true",
        help="Plot max across seeds for eval success rate (with SD band below)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Only keep eval points at multiples of this epoch (e.g. 2 to match every-2-epoch runs)",
    )
    args = parser.parse_args()

    # Collect run directories
    run_dirs = list(args.runs) if args.runs else []
    if args.glob:
        base = "checkpoints"
        matches = sorted(globmod.glob(os.path.join(base, args.glob)))
        run_dirs.extend(matches)

    if not run_dirs:
        parser.error("No run directories specified. Use positional args or --glob.")

    # Load all data
    configs = []
    all_data = []
    for rd in run_dirs:
        config, data = load_metrics(rd)
        if data is not None:
            configs.append(config)
            all_data.append(data)

    if not all_data:
        print("No valid metrics found.", file=sys.stderr)
        sys.exit(1)

    # Subsample eval epochs if requested
    if args.eval_every:
        step = args.eval_every
        for data in all_data:
            data["eval"] = {e: v for e, v in data["eval"].items() if e % step == 1}

    # Default output name
    if args.output is None:
        safe_title = args.title.lower().replace(" ", "_").replace(":", "")
        args.output = f"plots/{safe_title}.png"
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.group_by_config:
        # Group by config for multi-seed plots
        grouped = defaultdict(list)
        for config, data in zip(configs, all_data):
            key = config_key(config)
            label = config_label(config)
            grouped[label].append(data)
        plot_multiseed(grouped, args.title, args.output, max_epoch=args.max_epoch, cummax=args.cummax)
    else:
        # Individual comparison
        if args.labels and len(args.labels) == len(all_data):
            labels = args.labels
        else:
            labels = [os.path.basename(rd) for rd in run_dirs[: len(all_data)]]
        plot_comparison(all_data, labels, args.title, args.output, max_epoch=args.max_epoch, smoothing=args.smoothing)


if __name__ == "__main__":
    main()
