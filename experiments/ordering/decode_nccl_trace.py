#!/usr/bin/env python
"""Decode the NCCL flight-recorder dumps written on a process-group timeout.

Usage:
    python experiments/ordering/decode_nccl_trace.py logs/arch/nccl_trace_<jobid>_rank*
    python experiments/ordering/decode_nccl_trace.py logs/arch/nccl_trace_35542109_rank --tail 12

What to look for. The epoch-boundary hang shows the ranks disagreeing about how
many collectives the boundary contains (rank 0 three ahead of ranks 1-3). Line
the ranks up by `seq_id` and find the first entry where the ops stop matching:
the `frames` of that entry name the Python call site that one rank executes and
the others do not. `state` is `completed` for collectives that finished,
`started`/`scheduled` for the one each rank is stuck in.
"""
from __future__ import annotations

import argparse
import glob
import pickle
import sys
from pathlib import Path


def _load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _entries(dump) -> list:
    if isinstance(dump, dict):
        for key in ("entries", "nccl_comm_entries"):
            if key in dump:
                return list(dump[key])
    if isinstance(dump, list):
        return dump
    return []


def _fmt_frames(entry, max_frames: int) -> str:
    frames = entry.get("frames") or []
    keep = []
    for fr in frames:
        name = fr.get("name", "?")
        filename = fr.get("filename", "?")
        line = fr.get("line", "?")
        # Torch-internal frames are noise; the interesting frame is ours.
        if "/site-packages/torch/" in str(filename) and len(keep) >= 1:
            continue
        keep.append(f"      {filename}:{line} in {name}")
        if len(keep) >= max_frames:
            break
    return "\n".join(keep) if keep else "      <no python frames recorded>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="dump files, or the rank-prefix they share")
    ap.add_argument("--tail", type=int, default=8, help="entries per rank to print")
    ap.add_argument("--max-frames", type=int, default=6)
    args = ap.parse_args()

    files: list[Path] = []
    for p in args.paths:
        hits = sorted(glob.glob(p)) or sorted(glob.glob(p + "*"))
        files.extend(Path(h) for h in hits)
    if not files:
        print(f"no dump files matched {args.paths}", file=sys.stderr)
        print(
            "The launcher writes them only when the process group times out with\n"
            "TORCH_NCCL_DUMP_ON_TIMEOUT=1 set (scripts/hpc/arch/deadlock_debug.sh).",
            file=sys.stderr,
        )
        return 1

    for path in files:
        try:
            entries = _entries(_load(path))
        except Exception as e:
            print(f"== {path.name}: UNREADABLE ({e})")
            continue
        print(f"\n== {path.name}: {len(entries)} recorded collectives "
              f"(showing last {min(args.tail, len(entries))})")
        for entry in entries[-args.tail:]:
            print(
                f"  seq={entry.get('seq_id')} "
                f"op={entry.get('profiling_name') or entry.get('op_name')} "
                f"state={entry.get('state')} "
                f"in={entry.get('input_sizes')} out={entry.get('output_sizes')} "
                f"pg={entry.get('process_group')}"
            )
            print(_fmt_frames(entry, args.max_frames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
