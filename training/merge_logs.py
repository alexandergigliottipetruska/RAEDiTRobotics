"""Merge fragmented metrics/log files in a checkpoint directory.

Called automatically on resume to consolidate previous runs into single files.
Can also be run standalone: python -m training.merge_logs checkpoints/v3_foo/

Strategy:
- All run_info lines kept (tagged with restart_index)
- Epoch overlap <=200: last-writer-wins (the run that continued further)
- Epoch overlap >200: warn but still merge (backup kept)
- On resume: data beyond start_epoch is backed up to metrics_backup_*.jsonl
"""

import glob
import json
import logging
import os
import shutil

log = logging.getLogger(__name__)

OVERLAP_WARN_THRESHOLD = 200


def merge_metrics(save_dir: str, start_epoch: int | None = None) -> str:
    """Merge metrics_*.jsonl files into metrics.jsonl.

    Args:
        save_dir: Checkpoint directory containing the files.
        start_epoch: If resuming, the epoch we're resuming from.
            Data beyond this epoch is backed up before merging.

    Returns:
        Path to the merged metrics.jsonl file.
    """
    merged_path = os.path.join(save_dir, "metrics.jsonl")
    timestamped = sorted(glob.glob(os.path.join(save_dir, "metrics_*.jsonl")))

    # Collect all sources (merged + timestamped)
    sources = []
    if os.path.isfile(merged_path):
        sources.append(merged_path)
    sources.extend(timestamped)

    if not sources:
        return merged_path

    # If only a single merged file exists and no timestamped files, nothing to do
    if sources == [merged_path] and start_epoch is None:
        return merged_path

    # Parse all files into runs
    runs = []  # (run_info_obj | None, [(epoch, kind, raw_line), ...], max_epoch)
    for src in sources:
        run_info = None
        entries = []
        with open(src) as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") == "run_info":
                    if run_info is None:
                        run_info = obj
                elif "epoch" in obj:
                    ep = obj["epoch"]
                    if "eval_success_rate" in obj:
                        kind = "eval"
                    elif "per_timestep_loss" in obj:
                        kind = "detail"
                    else:
                        kind = "train"
                    entries.append((ep, kind, raw))
        max_ep = max((e[0] for e in entries), default=-1)
        runs.append((run_info, entries, max_ep))

    # Check for large overlaps (warn only)
    for i in range(1, len(runs)):
        prev_max = runs[i - 1][2]
        curr_epochs = [e[0] for e in runs[i][1]]
        if not curr_epochs:
            continue
        curr_min = min(curr_epochs)
        overlap = prev_max - curr_min + 1 if prev_max >= curr_min else 0
        if overlap > OVERLAP_WARN_THRESHOLD:
            log.warning(
                "Large epoch overlap (%d) between run %d (max epoch %d) and "
                "run %d (min epoch %d) in %s",
                overlap, i - 1, prev_max, i, curr_min, save_dir,
            )

    # Collect run_infos and epoch data (last-writer-wins)
    run_infos = []
    epoch_data = {}  # (epoch, kind) -> raw_line

    for idx, (ri, entries, _) in enumerate(runs):
        if ri is not None:
            ri["restart_index"] = idx
            run_infos.append(json.dumps(ri, default=str))
        for ep, kind, raw in entries:
            epoch_data[(ep, kind)] = raw

    # If resuming: backup data beyond start_epoch
    if start_epoch is not None:
        beyond = {k: v for k, v in epoch_data.items() if k[0] >= start_epoch}
        if beyond:
            backup_path = os.path.join(
                save_dir, f"metrics_backup_epoch{start_epoch}.jsonl"
            )
            # Don't overwrite existing backup
            if not os.path.isfile(backup_path):
                with open(backup_path, "w") as f:
                    for key in sorted(beyond.keys()):
                        f.write(beyond[key] + "\n")
                log.info(
                    "Backed up %d entries beyond epoch %d to %s",
                    len(beyond), start_epoch, os.path.basename(backup_path),
                )
            # Remove the beyond-epoch data from the merge
            for k in beyond:
                del epoch_data[k]

    # Build merged output
    merged_lines = list(run_infos)
    for key in sorted(epoch_data.keys()):
        merged_lines.append(epoch_data[key])

    # Write to temp then rename for atomicity
    tmp_path = merged_path + ".tmp"
    with open(tmp_path, "w") as f:
        for line in merged_lines:
            f.write(line + "\n")
    os.replace(tmp_path, merged_path)

    # Remove old timestamped files
    for src in timestamped:
        os.remove(src)

    n_epochs = len([k for k in epoch_data if k[1] == "train"])
    log.info(
        "Merged %d run(s), ~%d epochs into %s",
        len(run_infos), n_epochs, os.path.basename(merged_path),
    )
    return merged_path


def merge_logs(save_dir: str) -> str:
    """Merge train_*.log files into train.log.

    Returns:
        Path to the merged train.log file.
    """
    merged_path = os.path.join(save_dir, "train.log")
    timestamped = sorted(glob.glob(os.path.join(save_dir, "train_*.log")))

    if not timestamped:
        return merged_path

    # Collect all sources
    sources = []
    if os.path.isfile(merged_path):
        sources.append(merged_path)
    sources.extend(timestamped)

    if sources == [merged_path]:
        return merged_path

    tmp_path = merged_path + ".tmp"
    with open(tmp_path, "w") as out:
        for i, src in enumerate(sources):
            if i > 0:
                out.write(f"\n{'=' * 60}\n")
                out.write(f"=== RESTART {i} ({os.path.basename(src)}) ===\n")
                out.write(f"{'=' * 60}\n\n")
            with open(src) as inp:
                out.write(inp.read())
    os.replace(tmp_path, merged_path)

    for src in timestamped:
        os.remove(src)

    log.info("Merged %d log file(s) into train.log", len(sources))
    return merged_path


def merge_all(save_dir: str, start_epoch: int | None = None):
    """Merge both metrics and log files in a checkpoint directory."""
    merge_metrics(save_dir, start_epoch=start_epoch)
    merge_logs(save_dir)


# --- Standalone CLI ---
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Merge checkpoint logs")
    parser.add_argument("dirs", nargs="+", help="Checkpoint directory(ies)")
    parser.add_argument("--start-epoch", type=int, default=None,
                        help="Backup data beyond this epoch (resume mode)")
    args = parser.parse_args()

    for d in args.dirs:
        if os.path.isdir(d):
            print(f"\n=== {d} ===")
            merge_all(d, start_epoch=args.start_epoch)
        else:
            print(f"SKIP (not a directory): {d}")
