"""CPU-only source-policy reproduction and extraction of completed owner-run A/B evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parents[4]


def pipeline_chunk(n, cap, stages, ratio=283, stream_min=1024):
    """Port of Prefill::pipeline_chunk; predictions, not native-engine measurements."""
    if stages < 2 or n <= 0 or cap <= 0:
        return cap
    r = min(ratio, 1048576) if ratio > 0 else 0
    if r == 0:
        k = (n + cap - 1) // cap
        last = n - (k - 1) * cap
        if k <= 1 or last < stream_min:
            return cap
        return min(cap, (((n + k - 1) // k + 255) // 256) * 256)

    def cost(c):
        return ((n + c - 1) // c) * r + n + (stages - 1) * (r + min(c, n))

    today = min(n, cap)
    best, best_cost = today, cost(today)
    k_min = (n + cap - 1) // cap
    k_max = max(k_min, min(64, (n + 511) // 512))
    for k in range(k_min, k_max + 1):
        c = min(cap, max(512, (((n + k - 1) // k + 255) // 256) * 256))
        v = cost(c)
        if v < best_cost:
            best, best_cost = c, v
    return best if best < today and 100 * best_cost < 97 * cost(today) else cap


def completed_arm(harness, name):
    directory = harness / 'ab' / name
    cfg = json.loads((directory / 'cfg.json').read_text())
    log = (directory / 'engine.log').read_text(errors='replace')
    owned = re.search(r'own prompt buffers \(~(\d+) MiB each, (\d+)-token chunks\)', log)
    first = re.search(r'strata generate: expert cache (\d+) slots, ([\d.]+) GiB', log)
    later = re.findall(r'CUDA(\d+) runs layers (\d+)-(\d+), expert cache (\d+) slots \(([\d.]+) GiB\)', log)
    replies = [json.loads(p.read_text()) for p in sorted(directory.glob('long[0-9]*.json'))]
    return {
        'source': 'completed coordinator/owner A/B logs; not run by this analysis script',
        'arm': name,
        'prefill_requested': cfg['args'][cfg['args'].index('--prefill') + 1],
        'layer_split': cfg['layer_split'],
        'startup_owned_estimate_mib': int(owned[1]) if owned else None,
        'startup_chunk_cap': int(owned[2]) if owned else None,
        'first_cache_slots': int(first[1]) if first else None,
        'first_cache_gib': float(first[2]) if first else None,
        'later_caches': [dict(device=int(d), layer_begin=int(lb), layer_end_inclusive=int(le),
                              slots=int(slots), gib=float(gib)) for d, lb, le, slots, gib in later],
        'observed_pipe_reduction_logs': re.findall(r'strata prefill: .*tokens in .*stages.*', log),
        'timings': [r['timings'] for r in replies],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--harness', type=Path, default=Path('/home/ronald_rump/inbox/agents/claude/strata-ab'))
    parser.add_argument('--arms', nargs='+', default=['x28', 'x28p4k', 'y26p4096'])
    parser.add_argument('--segment-lengths', nargs='+', type=int, default=[16000, 16384, 36410, 100000, 143000])
    args = parser.parse_args()
    source_path = ROOT / 'src/prefill/prefill.cpp'
    source = source_path.read_text()
    assert 'return k * r + n + S1 * (r + std::min(c, n));' in source
    assert '100 * best_cost < 97 * today_cost' in source
    assert [pipeline_chunk(36410, c, 2) for c in (2048, 4096, 8192, 16384)] == [2048, 4096, 3328, 3328]
    predictions = []
    for n in args.segment_lengths:
        for cap in (2048, 4096, 8192, 16384):
            chunk = pipeline_chunk(n, cap, 2)
            count = (n + chunk - 1) // chunk
            predictions.append(dict(segment_tokens=n, cap=cap, predicted_chunk=chunk,
                                    predicted_chunks=count, predicted_tail=n - (count - 1) * chunk))
    profile_path = Path('/home/ronald_rump/Strata_donttouch/data/expert-profile.bin')
    data = profile_path.read_bytes()
    magic, version, nl, ne, slots, nr = struct.unpack_from('<4s5I', data)
    table = struct.unpack_from('<' + 'i' * (nl * ne), data, 24 + nr * 4)
    output = {
        'source_path': str(source_path),
        'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'prediction_scope': 'Python port of integer planner at b/a=283, two stages. Not a native-engine test. '
                            'Plans apply per checkpoint-delimited batched segment, not necessarily whole prompt; '
                            'the final prompt token ordinarily goes through the first decode window.',
        'planner_predictions': predictions,
        'completed_owner_arms': [completed_arm(args.harness, arm) for arm in args.arms],
        'live_profile': dict(path=str(profile_path), magic=magic.decode(), version=version, layers=nl,
                             experts=ne, slots=slots, ranked=nr, bytes=len(data),
                             trailing_table_is_rank_permutation=set(table) == set(range(nl * ne))),
    }
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
