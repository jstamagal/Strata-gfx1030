"""A/B q36 decode switches, exact output tokens and optional kernel references.

Runs on the GPU owner machine. Each case loads the model once. Prefixes of
length 1..8 exercise decode-width prompt calls; full prompts exercise prefill.
Use --args for --model, --layer-split, --mtp, etc. Logs remain in --out.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex

from q36_harness import Eng, load_prompts

SWITCHES = {
    "mmvq": "STRATA_Q36_RDNA2_MMVQ",
    "wide": "STRATA_Q36_MOE_OUT_WIDE",
    "fuse": "STRATA_Q36_FUSE",
    "experts": "STRATA_Q36_FUSE_EXPERT",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", default="build/strata-q36")
    ap.add_argument("--args", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--case", choices=[*SWITCHES, "all"], default="all")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--prompts", default=str(Path(__file__).with_name("parity_prompts.ids")))
    a = ap.parse_args()
    if a.n < 2:
        ap.error("--n must be at least 2 to exercise single-row decode")
    prompts = load_prompts(a.prompts)
    if not prompts or len(prompts[0]) < 8:
        ap.error("first prompt must contain at least 8 tokens")
    prompts = [prompts[0][:t] for t in range(1, 9)] + prompts
    a.out.mkdir(parents=True, exist_ok=True)
    args = shlex.split(a.args)
    if a.verify:
        args += ["--no-graph"]
    cases = ["baseline", *(list(SWITCHES) + ["combined"] if a.case == "all" else [a.case])]
    baseline = None
    for case in cases:
        env = {name: "0" for name in SWITCHES.values()}
        if case == "combined":
            env = {name: "1" for name in SWITCHES.values()}
        elif case in SWITCHES:
            env[SWITCHES[case]] = "1"
        # The engine's legacy verifier treats presence as on, even value "0".
        os.environ.pop("STRATA_Q36_VERIFY", None)
        if a.verify:
            env["STRATA_Q36_VERIFY"] = "1"
        log = a.out / (case + ".log")
        eng = Eng(a.exe, args, env=env, log=str(log))
        results = []
        try:
            for i, ids in enumerate(prompts):
                tokens, done, _, wall = eng.solo(ids, a.n)
                results.append({"tokens": tokens, "done": done, "wall_s": wall})
                print(case, i, done, flush=True)
        finally:
            eng.close()
            eng.logf.close()
        (a.out / (case + ".json")).write_text(json.dumps(results, indent=2) + "\n")
        if eng.p.returncode != 0:
            raise SystemExit(f"{case}: engine exit {eng.p.returncode}; see {log}")
        if a.verify:
            text = log.read_text()
            report = re.search(r"VERIFY: (\d+) comparisons, (\d+) mismatches", text)
            if not report or int(report[1]) == 0 or int(report[2]) != 0 or "VERIFY MISMATCH" in text:
                raise SystemExit(f"{case}: verifier failed or did not run; see {log}")
            if case != "baseline":
                for name, value in env.items():
                    if name in SWITCHES.values() and value == "1" and name + "=1" not in text:
                        raise SystemExit(f"{case}: switch missing in startup log: {name}")
                needed = {
                    "mmvq": ["rdna2 q8"],
                    "wide": ["moe_out q8"],
                    "fuse": ["fuse QKV", "fuse Q", "fuse shared q8", "fuse input norm"],
                    "experts": ["rdna2 experts"],
                }
                tags = sum(needed.values(), []) if case == "combined" else needed[case]
                for tag in tags:
                    if not re.search(r"VERIFY checked " + re.escape(tag) + r": [1-9]\d* comparisons", text):
                        raise SystemExit(f"{case}: no coverage of {tag}; check shapes/device/placement in {log}")
        seq = [r["tokens"] for r in results]
        if baseline is None:
            baseline = seq
        elif seq != baseline:
            raise SystemExit(f"{case}: tokens differ from baseline; compare JSON files in {a.out}")
        print(case + ": exact token match" if case != "baseline" else "baseline saved", flush=True)


if __name__ == "__main__":
    main()
