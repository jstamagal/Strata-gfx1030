"""GPU gate: identical greedy token IDs for GGUF, pack, profile, and pack+profile.

Each variant starts a fresh engine and receives the same sequence of prompts.
Logs include load time, prompt/decode time and swaps. Use a held-out corpus for
profile performance, and omit --verify for timings. Does not claim GPU parity
until run on the target hardware.
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect_profile_traces import load_tokenizer
from qwen35.q36_harness import Eng


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--engine', default='build/strata-q36')
    ap.add_argument('--model', required=True)
    ap.add_argument('--pack', required=True, type=Path)
    ap.add_argument('--profile', required=True, type=Path)
    ap.add_argument('--corpus', required=True, type=Path, help='JSON list of {"p": "prompt text"}')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--mtp', type=int, default=1)
    ap.add_argument('--layer-split', default='auto')
    ap.add_argument('--expert-slots', type=int, default=0, help='single-device forced hybrid coverage')
    ap.add_argument('--chunk', type=int, default=2048)
    ap.add_argument('--max-context', type=int, default=32768)
    ap.add_argument('--max-new', type=int, default=128)
    ap.add_argument('--verify', action='store_true')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    corpus = json.loads(a.corpus.read_text())
    if not corpus:
        ap.error('corpus is empty')
    tok = load_tokenizer(a.pack / 'tokenizer')
    prompts = [tok.encode(f"<|im_start|>user\n{c['p']}<|im_end|>\n"
                          '<|im_start|>assistant\n<think>\n\n</think>\n\n', parse_special=True) for c in corpus]
    args = ['--model', a.model, '--mtp', str(a.mtp), '--layer-split', a.layer_split,
            '--chunk', str(a.chunk), '--max-context', str(a.max_context)]
    if a.expert_slots:
        args += ['--expert-slots', str(a.expert_slots)]
    variants = {'gguf': [], 'pack': ['--pack', str(a.pack)],
                'profile': ['--expert-profile', str(a.profile)],
                'both': ['--pack', str(a.pack), '--expert-profile', str(a.profile)]}
    reference = None
    report = {}
    os.environ.pop('STRATA_Q36_VERIFY', None)
    for name, extra in variants.items():
        # An inherited pack/verify environment must not silently contaminate the baseline.
        eng = Eng(a.engine, args + extra, env={'STRATA_Q36_PACK': '', **({'STRATA_Q36_VERIFY': '1'} if a.verify else {})},
                  log=str(a.out / f'{name}.log'))
        tokens, metrics = [], []
        try:
            for ids in prompts:
                generated, done, ttft, elapsed = eng.solo(ids, a.max_new, keys='temperature=0')
                tokens.append(generated)
                metrics.append(dict(done=done, ttft_s=ttft, elapsed_s=elapsed))
            eng.p.stdin.close()
            if eng.p.wait(timeout=120) != 0:
                raise RuntimeError(f'{name}: engine failed; see log')
        finally:
            eng.close()
        (a.out / f'{name}.tokens.json').write_text(json.dumps(tokens))
        report[name] = dict(load_s=eng.load_s, prompts=metrics)
        (a.out / 'timings.json').write_text(json.dumps(report, indent=2))
        if reference is None:
            reference = tokens
        elif tokens != reference:
            raise SystemExit(f'FAIL: {name} greedy token IDs differ from GGUF; see {a.out}')
        if a.verify:
            import re
            log = (a.out / f'{name}.log').read_text()
            summary = re.search(r'VERIFY: (\d+) comparisons, (\d+) mismatches', log)
            if not summary or int(summary[1]) == 0 or int(summary[2]) != 0:
                raise SystemExit(f'FAIL: {name} missing/failed kernel verification; see log')
        print(f'{name}: greedy tokens match; load {eng.load_s:.2f}s; logs in {a.out}', flush=True)
    print('PASS: all four variants produced identical greedy token IDs')


if __name__ == '__main__':
    main()
