"""Collect real q36 router counts over a chat corpus, using one engine startup.

Uses the existing Strata tokenizer/corpus and q36 serving harness. --out is one
cumulative trace; feed it once to q36_profile.py. Trace runs are not speed runs.
"""
import argparse
import json
from pathlib import Path
from collect_profile_traces import load_tokenizer
from qwen35.q36_harness import Eng


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--engine', default='build/strata-q36')
    ap.add_argument('--model', required=True)
    ap.add_argument('--pack', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--corpus', type=Path, default=Path(__file__).resolve().parent.parent / 'bench/profile_corpus.json')
    ap.add_argument('--max-new', type=int, default=250)
    ap.add_argument('--mtp', type=int, default=1)
    ap.add_argument('--layer-split', default='auto')
    ap.add_argument('--max-context', type=int, default=32768)
    ap.add_argument('--chunk', type=int, default=2048)
    a = ap.parse_args()
    if a.out.exists():
        ap.error('--out already exists; use a new trace path')
    corpus = json.loads(a.corpus.read_text())
    if not corpus:
        ap.error('corpus is empty')
    tok = load_tokenizer(a.pack / 'tokenizer')
    eng = Eng(a.engine, ['--model', a.model, '--pack', str(a.pack), '--trace-experts', str(a.out),
                         '--mtp', str(a.mtp), '--layer-split', a.layer_split, '--max-context', str(a.max_context),
                         '--chunk', str(a.chunk)], log=str(a.out) + '.log')
    try:
        for i, item in enumerate(corpus):
            # Same explicit non-thinking Qwen chat framing as collect_profile_traces.py.
            ids = tok.encode(f"<|im_start|>user\n{item['p']}<|im_end|>\n"
                             '<|im_start|>assistant\n<think>\n\n</think>\n\n', parse_special=True)
            tokens, done, _, _ = eng.solo(ids, a.max_new)
            print(f'{i+1}/{len(corpus)}: {len(ids)} prompt, {len(tokens)} generated; {done}', flush=True)
        # EOF exits the server normally and publishes the final MTP counts.
        eng.p.stdin.close()
        if eng.p.wait(timeout=120) != 0:
            raise RuntimeError('engine failed; see trace log')
    finally:
        eng.close()
    print(f'counts: {a.out}; freeze with tools/q36_profile.py')


if __name__ == '__main__':
    main()
