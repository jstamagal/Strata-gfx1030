"""Merge q36 Q36C routing-count snapshots into the main engine's STRP v1 format.

Each input must be a separate run, not successive snapshots of the same run.
Counts include all executed router rows, including speculative drafts/verifies
that may later be discarded; warmup is excluded. No synthetic benchmark data.
"""
import argparse
from pathlib import Path
import struct
from make_profile import write_profile


def read_counts(path):
    raw = Path(path).read_bytes()
    if len(raw) < 16 or raw[:4] != b'Q36C':
        raise ValueError(f'{path}: not a q36 count trace')
    version, nl, ne = struct.unpack_from('<3I', raw, 4)
    if version != 1 or not 0 < nl <= 65536 or not 0 < ne <= 65536 or len(raw) != 16 + nl * ne * 8:
        raise ValueError(f'{path}: invalid version, geometry or length')
    return nl, ne, struct.unpack_from(f'<{nl*ne}Q', raw, 16)


def freeze(paths, out):
    nl, ne, first = read_counts(paths[0])
    counts = list(first)
    for path in paths[1:]:
        l, e, values = read_counts(path)
        if (l, e) != (nl, ne):
            raise ValueError('traces have different geometries')
        counts = [a+b for a, b in zip(counts, values)]
    if not any(counts):
        raise ValueError('no routed entries in these traces')
    ranked = sorted(((l, e) for l in range(nl) for e in range(ne)),
                    key=lambda p: (-counts[p[0]*ne+p[1]], p))
    write_profile(out, ranked, ne, nl)
    print(f'wrote {out}: {nl}x{ne}, {sum(counts)} routed entries; all pairs ranked')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('traces', nargs='+', type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    freeze(args.traces, args.out)


if __name__ == '__main__':
    main()
