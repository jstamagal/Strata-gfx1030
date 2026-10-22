"""Direct q36 sampled-chat bench: fixed tokenized prompts, engine decode timing and draft counts.

Example: python tools/qwen35/q36_mtp_sampled_bench.py --exe build/bin/strata-q36 \
    --args "--model /path/to/first.gguf --tensor-split --mtp 3 --draft-vocab 65536" \
    --out /tmp/q36-sampled.json --log /tmp/q36-sampled.log
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shlex
import statistics

from q36_harness import Eng, load_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--args", required=True)
    ap.add_argument("--prompts", default=str(Path(__file__).with_name("parity_prompts.ids")))
    ap.add_argument("--count", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    prompts = load_prompts(a.prompts)[:a.count]
    if not prompts or a.rounds < 1 or a.n < 1:
        ap.error("need at least one prompt, one round and one output token")
    eng = Eng(a.exe, shlex.split(a.args), log=a.log)
    rows = []
    try:
        for rd in range(a.rounds):
            for i, prompt in enumerate(prompts):
                keys = f"temperature=1 top_p=0.95 top_k=20 seed={a.seed}"
                tokens, done, _, _ = eng.solo(prompt, a.n, keys)
                fields = done.split()
                if len(fields) < 8:
                    raise RuntimeError(f"short DONE line: {done}")
                decode_ms = float(fields[4])
                accepted, offered = int(fields[6]), int(fields[7])
                row = {"round": rd, "prompt": i, "tokens": len(tokens), "decode_ms": decode_ms,
                       "tok_s": 1000.0 * len(tokens) / decode_ms if decode_ms > 0 else 0.0,
                       "accepted": accepted, "offered": offered,
                       "token_sha256": hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest()}
                rows.append(row)
                print(f"round {rd + 1} prompt {i + 1}: {len(tokens)} tokens, {row['tok_s']:.1f} tok/s, "
                      f"{accepted}/{offered} drafts accepted", flush=True)
    finally:
        eng.close()
    log_text = Path(a.log).read_text(errors="replace")
    stats = [{"steps": int(m.group(1)), "tokens_per_step": float(m.group(2)),
              "verify_rows_per_step": float(m.group(3))}
             for m in re.finditer(r"MTP request: (\d+) steps, ([\d.]+) tokens/step, ([\d.]+) verify rows/step", log_text)]
    lookups = [{"steps": int(m.group(1)), "tokens_per_step": float(m.group(2))}
               for m in re.finditer(r"MTP request lookup: (\d+) steps, ([\d.]+) tokens/step", log_text)]
    if len(stats) == len(rows):
        for row, stat in zip(rows, stats):
            row.update(stat)
            row["ms_per_step"] = row["decode_ms"] / stat["steps"] if stat["steps"] else None
    cost = [{"block_calls": int(m.group(1)), "block_ms": float(m.group(2)),
             "head_calls": int(m.group(3)), "head_ms": float(m.group(4))}
            for m in re.finditer(r"MTP_COST: (\d+) block calls, ([\d.]+) ms/block call; "
                                 r"(\d+) draft heads, ([\d.]+) ms/draft head", log_text)]
    stages = [{"input_ms": float(m.group(1)), "mixer_ms": float(m.group(2)), "moe_ms": float(m.group(3))}
              for m in re.finditer(r"MTP_STAGE: input\+eh_proj ([\d.]+) ms/call, "
                                   r"attention\+mixer ([\d.]+) ms/call, MoE\+shared ([\d.]+) ms/call", log_text)]
    t1_stages = [{"calls": int(m.group(1)), "input_ms": float(m.group(2)),
                  "mixer_ms": float(m.group(3)), "moe_ms": float(m.group(4)), "head_ms": float(m.group(5))}
                 for m in re.finditer(r"MTP_STAGE_T1: (\d+) calls; input\+eh_proj ([\d.]+) ms/call, "
                                      r"attention\+mixer ([\d.]+) ms/call, MoE\+shared ([\d.]+) ms/call, "
                                      r"draft head ([\d.]+) ms/call", log_text)]
    stage_share = None
    if cost and stages and stats and sum(s["steps"] for s in stats):
        steps = sum(s["steps"] for s in stats)
        c, s = cost[-1], stages[-1]
        stage_share = {"block_calls_per_step": c["block_calls"] / steps,
                       "head_calls_per_step": c["head_calls"] / steps,
                       "mtp_ms_per_step": ((s["input_ms"] + s["mixer_ms"] + s["moe_ms"]) * c["block_calls"]
                                           + c["head_ms"] * c["head_calls"]) / steps}
        costs = [r["ms_per_step"] for r in rows if r.get("ms_per_step")]
        if costs:
            stage_share["percent_of_diagnostic_step"] = 100 * stage_share["mtp_ms_per_step"] / statistics.median(costs)
    Path(a.out).write_text(json.dumps({"settings": {"n": a.n, "seed": a.seed,
                                       "temperature": 1.0, "top_p": 0.95, "top_k": 20},
                                   "rows": rows, "mtp_stats": stats, "lookup_stats": lookups,
                                   "mtp_cost": cost, "mtp_stages": stages, "mtp_t1_stages": t1_stages,
                                   "mtp_stage_share": stage_share}, indent=2) + "\n")
    print(f"median {statistics.median(r['tok_s'] for r in rows):.1f} tok/s; "
          f"drafts {sum(r['accepted'] for r in rows)}/{sum(r['offered'] for r in rows)} accepted")
    if stats:
        print(f"mean {statistics.mean(s['tokens_per_step'] for s in stats):.3f} tokens/step "
              f"over {len(stats)} requests")
        costs = [r["ms_per_step"] for r in rows if r.get("ms_per_step") is not None]
        if costs:
            print(f"median {statistics.median(costs):.3f} ms/step (request decode time / MTP steps)")
    if cost:
        print(f"MTP block {cost[-1]['block_ms']:.3f} ms/call, head {cost[-1]['head_ms']:.3f} ms/draft head "
              "(diagnostic mode)")
    if lookups:
        print(f"lookup: {sum(s['steps'] for s in lookups)} steps, "
              f"{statistics.mean(s['tokens_per_step'] for s in lookups):.3f} tokens/step")
    if stages:
        s = stages[-1]
        print(f"MTP stages: input {s['input_ms']:.3f}, mixer {s['mixer_ms']:.3f}, "
              f"MoE {s['moe_ms']:.3f} ms/call (diagnostic mode)")
    if t1_stages:
        s = t1_stages[-1]
        print(f"MTP T=1: {s['calls']} calls; input {s['input_ms']:.3f}, "
              f"mixer {s['mixer_ms']:.3f}, MoE {s['moe_ms']:.3f}, "
              f"head {s['head_ms']:.3f} ms/call (diagnostic mode)")
    if stage_share:
        print(f"MTP {stage_share['mtp_ms_per_step']:.3f} ms/step from "
              f"{stage_share['block_calls_per_step']:.2f} block calls and "
              f"{stage_share['head_calls_per_step']:.2f} draft heads per step; "
              f"{stage_share.get('percent_of_diagnostic_step', 0):.1f}% of diagnostic step (diagnostic mode)")


if __name__ == "__main__":
    main()
