#!/usr/bin/env python3
"""Pair STRATA_Q36_TPX_DENSE_TRACE=1's T=1 shards with a rocprof summary.

The profile reports one mean per kernel class, so GB/s is an estimate when
several matrix shapes share a class. Use hip_mmvq_row_parity --tpx-trace for
per-shape timings and bitwise checks.
"""
import argparse
import collections
import re


def read_profile(path):
    timings = {}
    for line in open(path, encoding="utf-8"):
        if line.startswith("GPU2"):
            break
        m = re.match(r"\s+[\d.]+ us/tok\s+[\d.]+/tok\s+([\d.]+)us\s+(.+)$", line)
        if m:
            timings[m.group(2).strip()] = float(m.group(1))
    return timings


def kernel_name(typ, k, route):
    if route == "q8_small":
        return "q36k::k_q8_rows<1, 1, false>" if k <= 1024 else None
    if route != "native_mmvq":
        return None
    if typ == 8:
        return f"native_small_mmvq_kernel<Q80Block, 8, {str(k // 32 < 32).lower()}>"
    if typ == 23:
        return f"native_iq4_xs_mmvq_kernel<{str(k // 256 < 16).lower()}>"
    if typ == 12:
        return f"native_q4_k_mmvq_kernel<{str(k // 256 < 8).lower()}>"
    if typ == 13:
        return f"native_q5_k_mmvq_kernel<{str(k // 256 < 8).lower()}>"
    if typ == 14:
        return f"native_q6_k_mmvq_kernel<{str(k // 256 < 8).lower()}>"
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", help="stderr log from STRATA_Q36_TPX_DENSE_TRACE=1")
    p.add_argument("profile", help="T=1 rocprof summary, e.g. int7-T1-breakdown.txt")
    p.add_argument("--rank", type=int, default=0)
    args = p.parse_args()
    timings = read_profile(args.profile)
    grouped = collections.Counter()
    for line in open(args.trace, encoding="utf-8"):
        if "TPX_DENSE rank=" not in line:
            continue
        fields = dict(re.findall(r"(rank|name|type|K|M|route|bytes)=([^\s]+)", line))
        if int(fields["rank"]) != args.rank:
            continue
        name = fields["name"]
        name = re.sub(r"^blk\.\d+\.", "blk.*.", name)
        key = (name, int(fields["type"]), int(fields["K"]), int(fields["M"]),
               fields["route"], int(fields["bytes"]))
        grouped[key] += 1
    rows = []
    for (name, typ, k, m, route, size), calls in grouped.items():
        klass = kernel_name(typ, k, route)
        us = timings.get(klass) if klass else None
        bw = size / us / 1000 if us else None
        rows.append((-(calls * us if us else 0), name, typ, k, m, route, calls, size, us, bw))
    print("rank name type K M route calls bytes/call class_us GB/s est_us/window")
    for _, name, typ, k, m, route, calls, size, us, bw in sorted(rows):
        print(f"{args.rank} {name} {typ} {k} {m} {route} {calls} {size} "
              f"{us if us else '-'} {round(bw) if bw else '-'} "
              f"{round(calls * us) if us else '-'}")
    print("Class means mix shapes; use the GPU lab's per-shape GB/s before choosing a switch.")


if __name__ == "__main__":
    main()
