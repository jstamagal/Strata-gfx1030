#!/usr/bin/env python3
"""Compare STRATA_Q36_DUMP_LAYERS traces (numpy required); stream one residual at a time.

Both runs must use the same tokens, forced continuation, chunk size and MTP setting.
Relative error is ||test-reference||_2 / ||reference||_2. The summary also reports
the largest relative error of any individual token row, so long prompts cannot hide it.
"""
import argparse
import struct
import sys
from collections import defaultdict

import numpy as np


def read_exact(f, count):
    data = f.read(count)
    if len(data) != count:
        raise ValueError(f"{f.name}: truncated at byte {f.tell()} (wanted {count}, got {len(data)})")
    return data


def header(f):
    if read_exact(f, 8) != b"Q36LYR1\0":
        raise ValueError(f"{f.name}: unsupported dump format")
    endian, width, layers = struct.unpack("<3I", read_exact(f, 12))
    if endian != 0x01020304 or not width or not layers:
        raise ValueError(f"{f.name}: invalid header")
    return width, layers


def record(f, width, layers):
    raw = f.read(32)
    if not raw:
        return None
    if len(raw) != 32:
        raise ValueError(f"{f.name}: truncated record header")
    layer, stage, rows, flags, call, p0 = struct.unpack("<4IQq", raw)
    if layer > layers or stage > 1 or not rows or flags & ~7 or bool(flags & 2) != (layer == layers):
        raise ValueError(f"{f.name}: invalid record header {raw.hex()}")
    meta = read_exact(f, 3 * rows * 4)
    x = np.frombuffer(read_exact(f, rows * width * 4), dtype="<f4").reshape(rows, width)
    if not np.isfinite(x).all():
        raise ValueError(f"{f.name}: nonfinite residual at call {call}, layer {layer}, stage {stage}")
    return (layer, stage, rows, flags, call, p0), meta, x


def relative(error2, reference2):
    return np.sqrt(error2 / reference2) if reference2 else (0.0 if not error2 else float("inf"))


def compare(args):
    totals = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
    first = None
    records = 0
    with open(args.reference, "rb") as a, open(args.test, "rb") as b:
        shape = header(a)
        if header(b) != shape:
            raise ValueError("dump model dimensions differ")
        while True:
            ra, rb = record(a, *shape), record(b, *shape)
            if ra is None or rb is None:
                if ra is not None or rb is not None:
                    raise ValueError(f"record counts differ after {records} records")
                break
            if ra[:2] != rb[:2]:
                raise ValueError(f"record {records}: window/token/cell/slot metadata differ: {ra[0]} vs {rb[0]}")
            layer, stage, rows, flags, call, p0 = ra[0]
            ref = ra[2].astype(np.float64)
            delta = rb[2].astype(np.float64) - ref
            err2 = np.sum(delta * delta, axis=1)
            ref2 = np.sum(ref * ref, axis=1)
            row_rel = np.array([relative(e, r) for e, r in zip(err2, ref2)])
            ma = float(np.max(np.abs(delta)))
            rel = relative(float(err2.sum()), float(ref2.sum()))
            worst = int(row_rel.argmax())
            point = "moe" if stage else "mixer"
            label = "mtp" if flags & 2 else str(layer)
            if first is None and row_rel[worst] > args.threshold:
                ids = np.frombuffer(ra[1], dtype="<i4").reshape(3, rows)
                first = (call, label, point, int(ids[1, worst]), int(ids[0, worst]), float(row_rel[worst]))
            if args.per_window:
                print(f"call={call} p0={p0} T={rows} layer={label} {point:5} "
                      f"max_abs={ma:.7g} rel_l2={rel:.7g} max_row_rel={row_rel[worst]:.7g}")
            s = totals[(layer, stage)]
            s[0] = max(s[0], ma)
            s[1] += float(err2.sum())
            s[2] += float(ref2.sum())
            s[3] = max(s[3], float(row_rel[worst]))
            s[4] += rows
            records += 1
    if not records:
        raise ValueError("dumps contain no residual records")
    print("layer stage      max_abs       rel_l2  max_row_rel     rows")
    for (layer, stage), (ma, err2, ref2, worst, rows) in sorted(totals.items()):
        label = "mtp" if layer == shape[1] else str(layer)
        print(f"{label:>5} {'moe' if stage else 'mixer':5} {ma:12.5g} {relative(err2, ref2):12.5g} {worst:12.5g} {rows:8}")
    if first:
        call, layer, point, cell, token, rel = first
        print(f"First record above row-relative threshold {args.threshold:g}: "
              f"call={call} layer={layer} {point} cell={cell} token={token} rel={rel:.7g}")
    else:
        print(f"All {records} residual records below row-relative threshold {args.threshold:g}.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference")
    ap.add_argument("test")
    ap.add_argument("--per-window", action="store_true")
    ap.add_argument("--threshold", type=float, default=1e-3, help="localization threshold, not a quality gate (default 1e-3)")
    args = ap.parse_args()
    try:
        compare(args)
    except (OSError, ValueError) as e:
        ap.exit(2, f"{e}\n")


if __name__ == "__main__":
    main()
