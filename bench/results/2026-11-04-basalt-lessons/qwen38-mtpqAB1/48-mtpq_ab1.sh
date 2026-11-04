#!/bin/bash
# mtpq A/B 1: MTP drafter weight variants (new rt dirs built from BF16) on the live IQ3_S config, agent AND edit
# prompts, two passes (the second in reverse arm order), then drafter-profile arms on agent prompts.
# Binary: strata-q38-mtpq build-hip/strata (main + native/mixed drafter experts + capture hook, capture off here).
# Every arm: STRATA_KV_PF_SLOTS=1 (live).  Gates per run: CUDA1 11776 expert slots, 100% resident, 4 zero-doorbell
# verify graphs.  Text: md5 per case against the live arm (exact verify: a changed draft changes only which tokens are
# verified together).
# GPTQ variants are calibrated on agent requests 0/2/4 only (warm-up, code_rust_committed_ring,
# json_multi_tool_result_merge): the other four agent cases and all edit cases are held out.
H=/home/ronald_rump/inbox/agents/claude/strata-ab
E=/home/ronald_rump/inbox/agents/claude/strata-q38-mtpq/build-hip/strata
EDIT=$H/edit-prompts.json
AGENT=/home/ronald_rump/inbox/agents/claude/strata-q38-dflash2/bench/results/2026-11-04-dflash2-agent-prompts.json
M=/home/ronald_rump/Strata-data/mtp
# name|rt dir   (the first is the control)
ARMS=(
    "live|$M/rt"
    "q2w|$M/q38-s2-q2w"
    "s2gptq|$M/q38-s2-gptq-a"
    "mixr15|$M/q38-mix-r15-gptq-a"
    "mixr40|$M/q38-mix-r40-gptq-a"
    "qo4|$M/q38-live-qo-q4_0"
)
# name|rt dir   drafter profile (STRATA_DRAFT_TIMING=1 STRATA_MTP_PROFILE=1 STRATA_PIPELINE_TRACE=pipeline.tsv), agent prompts, one pass
PROF=(
    "live|$M/rt"
    "q4n|$M/q38-native-q4_0-rtn"
    "mixr40|$M/q38-mix-r40-gptq-a"
    "n2live|$M/q38-native-q2_0-live"
)
# STRATA_HIP_MTP_PRIORITY=0 profile arms (read by the mtpq agent; not in the summary)
NOPRIO=("live|$M/rt" "q4n|$M/q38-native-q4_0-rtn")
cd "$H" || exit 1
for a in "${NOPRIO[@]}"; do [ -e "$H/ab/mtpq1_prof_${a%%|*}_noprio" ] && { echo "GATE FAIL: mtpq1_prof_${a%%|*}_noprio exists"; exit 1; }; done
[ -x "$E" ] || { echo "GATE FAIL: $E missing"; exit 1; }
sha256sum "$E"
for a in "${ARMS[@]}" "${PROF[@]}"; do
    d=${a#*|}
    [ -f "$d/experts.bin" ] && [ -f "$d/dense.bin" ] || { echo "GATE FAIL: $d incomplete"; exit 1; }
done
for a in "${ARMS[@]}"; do for s in agent edit; do for p in p1 p2; do
    [ -e "$H/ab/mtpq1_${a%%|*}_${s}_$p" ] && { echo "GATE FAIL: $H/ab/mtpq1_${a%%|*}_${s}_$p exists"; exit 1; }
done; done; done
for a in "${PROF[@]}"; do [ -e "$H/ab/mtpq1_prof_${a%%|*}" ] && { echo "GATE FAIL: $H/ab/mtpq1_prof_${a%%|*} exists"; exit 1; }; done
unset STRATA_MTP_CAPTURE STRATA_DRAFT_TIMING STRATA_MTP_PROFILE STRATA_PIPELINE_TRACE
run() {
    local name=$1 prompts=$2 dir=$3
    shift 3
    env CFG="$H/qwen-iq3s.json" MODEL=qwen3.8-abl-iq3_s PROMPTS="$prompts" "$H/agab2.sh" "$name" "EXE=$E" \
        "STRATA_KV_PF_SLOTS=1" "ARG=--mtp $dir" "$@"
    echo "HARNESS_STATUS $name $?"
}
echo "mtpq_ab1 start $(date +%H:%M)"
for a in "${ARMS[@]}"; do run "mtpq1_${a%%|*}_agent_p1" "$AGENT" "${a#*|}"; run "mtpq1_${a%%|*}_edit_p1" "$EDIT" "${a#*|}"; done
for (( i=${#ARMS[@]}-1; i>=0; i-- )); do a=${ARMS[$i]}
    run "mtpq1_${a%%|*}_edit_p2" "$EDIT" "${a#*|}"; run "mtpq1_${a%%|*}_agent_p2" "$AGENT" "${a#*|}"; done
for a in "${PROF[@]}"; do run "mtpq1_prof_${a%%|*}" "$AGENT" "${a#*|}" "STRATA_DRAFT_TIMING=1" "STRATA_MTP_PROFILE=1" "STRATA_PIPELINE_DEBUG=1" "STRATA_PIPELINE_TRACE=pipeline.tsv"; done
for a in "${NOPRIO[@]}"; do run "mtpq1_prof_${a%%|*}_noprio" "$AGENT" "${a#*|}" "STRATA_DRAFT_TIMING=1" "STRATA_MTP_PROFILE=1" "STRATA_PIPELINE_DEBUG=1" "STRATA_PIPELINE_TRACE=pipeline.tsv" "STRATA_HIP_MTP_PRIORITY=0"; done
python3 -I - "$H/ab" "${#ARMS[@]}" "${ARMS[@]}" "${PROF[@]}" <<'PY'
import glob, hashlib, json, os, re, sys
root = sys.argv[1]; na = int(sys.argv[2])
arms = [a.split("|", 1)[0] for a in sys.argv[3: 3 + na]]
prof = [a.split("|", 1)[0] for a in sys.argv[3 + na:]]
HELD = {"code_async_batcher", "code_typescript_tool_dispatch", "json_parallel_tool_plan", "json_patch_tool_payload"}
def run(name):
    D = f"{root}/{name}"
    if not os.path.isdir(D):
        return None
    cases = {}
    for f in sorted(glob.glob(D + "/out_*.json")):
        cid = os.path.basename(f)[4:-5]
        try:
            d = json.load(open(f)); t = d["timings"]; txt = d["choices"][0]["message"]["content"]
        except Exception as e:
            cases[cid] = {"error": str(e)}; continue
        n = t["predicted_n"]; ms = n / t["predicted_per_second"] * 1000; acc = t.get("draft_n_accepted") or 0
        cases[cid] = {"n": n, "ms": ms, "acc": acc, "dr": t.get("draft_n") or 0, "md5": hashlib.md5(txt.encode()).hexdigest()}
    log = open(D + "/engine.log", errors="replace").read() if os.path.exists(D + "/engine.log") else ""
    gate = {"slots": re.findall(r"CUDA1 runs layers \S+, expert cache (\d+) slots", log), "zd": log.count("zero-doorbell graph"),
            "resident": "100% of the experts resident" in log,
            "mtp_mib": re.findall(r"draft layer loaded, (\d+) MiB of VRAM", log),
            "cuda1_free": re.findall(r"CUDA1: (\d+) MiB of VRAM free with everything loaded", log),
            "timing": [l for l in log.splitlines() if "draft timing" in l or "mtp profile" in l][-4:]}
    gate["ok"] = gate["slots"] == ["11776"] and gate["zd"] == 4 and gate["resident"] and not any("error" in c for c in cases.values())
    return cases, gate
def agg(cases, keep=lambda c: True):
    c = [v for k, v in cases.items() if keep(k) and "error" not in v]
    n = sum(x["n"] for x in c); ms = sum(x["ms"] for x in c); acc = sum(x["acc"] for x in c); dr = sum(x["dr"] for x in c)
    r = n - acc
    return {"tps": n / ms * 1000 if ms else 0, "ms_rd": ms / r if r else 0, "tok_rd": n / r if r else 0, "acc": acc, "dr": dr}
res = {}
for s in ("agent", "edit"):
    print(f"\n== {s}: both passes pooled (tok/s, accepted, ms/round, tokens/round); per pass tok/s; gates")
    base = None
    for a in arms:
        rs = [run(f"mtpq1_{a}_{s}_{p}") for p in ("p1", "p2")]
        rs = [r for r in rs if r]
        if not rs:
            print(f"{a:24s} MISSING"); continue
        pooled = {}
        for i, (cases, g) in enumerate(rs):
            for k, v in cases.items():
                pooled[f"{k}#{i}"] = v
        A = agg(pooled)
        Ah = agg(pooled, lambda k: k.split("#")[0] in HELD) if s == "agent" else None
        Ac = agg(pooled, lambda k: k.split("#")[0] not in HELD) if s == "agent" else None
        if base is None:
            base = (A, Ah, Ac, rs[0][0])
        same = sum(1 for k, v in rs[0][0].items() if "md5" in v and base[3].get(k, {}).get("md5") == v["md5"])
        rel = lambda x, b: (x / b - 1) * 100 if b else 0
        line = (f"{a:24s} {A['tps']:7.2f} tok/s ({rel(A['tps'], base[0]['tps']):+5.1f}%)  accepted {A['acc']:5d} "
                f"({rel(A['acc'], base[0]['acc']):+5.1f}%)  {A['ms_rd']:6.2f} ms/rd ({rel(A['ms_rd'], base[0]['ms_rd']):+5.1f}%)  "
                f"{A['tok_rd']:5.3f} tok/rd ({rel(A['tok_rd'], base[0]['tok_rd']):+5.1f}%)  passes "
                + " ".join(f"{agg(r[0])['tps']:.1f}" for r in rs) + f"  text {same}/{len(rs[0][0])} same  gates "
                + " ".join(("OK" if r[1]["ok"] else f"FAIL{r[1]['slots']}/zd{r[1]['zd']}/res{r[1]['resident']}") for r in rs)
                + f"  drafter {rs[0][1]['mtp_mib']} MiB, CUDA1 free {rs[0][1]['cuda1_free']}")
        if Ah:
            line += (f"\n{'':24s} held-out cases {Ah['tps']:7.2f} tok/s ({rel(Ah['tps'], base[1]['tps']):+5.1f}%), "
                     f"calibration cases {Ac['tps']:7.2f} tok/s ({rel(Ac['tps'], base[2]['tps']):+5.1f}%)")
        print(line)
print("\n== drafter profile arms (agent, one pass)")
for a in prof:
    r = run(f"mtpq1_prof_{a}")
    if not r:
        print(f"{a}: MISSING"); continue
    A = agg(r[0])
    print(f"{a}: {A['tps']:.2f} tok/s, {A['ms_rd']:.2f} ms/rd, {A['tok_rd']:.3f} tok/rd, gates {'OK' if r[1]['ok'] else 'FAIL'}")
    for l in r[1]["timing"]:
        print("   ", l)
PY
echo "mtpq_ab1 end $(date +%H:%M)"
