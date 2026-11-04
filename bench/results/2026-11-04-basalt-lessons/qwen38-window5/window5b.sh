#!/bin/bash
# window 5b (after window 5): exact Q5_1 TJ16 prompt tile, non-exact Q8K switches with first-token logits, IQ prompt clamp
until grep -q 'WINDOW5 DONE' /tmp/basalt-prof/window5.out 2>/dev/null; do sleep 15; done
echo "window 5b start $(date +%H:%M)"
cd /tmp/basalt-prof
C=EXE=/tmp/basalt-prof/strata-rdna2-control
K=EXE=/tmp/basalt-prof/strata-rdna2-prompt-clamp
N=EXE=/tmp/basalt-prof/strata-rdna2-nl-perm
F="ARG=--adapt-swaps 0"
L=/tmp/basalt-prof/logits
{
export MAXTOK=300 RUNS=2
CFG=/tmp/basalt-prof/qwen-ud4.json ./pab.sh k5_ctl "$C" "$F" CFG.layer_split=26 "ARG=--prefill 4096" STRATA_DUMP_FIRST_LOGITS=$L/k5_ctl
CFG=/tmp/basalt-prof/qwen-ud4.json ./pab.sh k5_tj16 "$C" "$F" CFG.layer_split=26 "ARG=--prefill 4096" STRATA_PF_Q51_TJ=16 STRATA_DUMP_FIRST_LOGITS=$L/k5_tj16
CFG=/tmp/basalt-prof/qwen-ud4.json ./pab.sh k5_q8k "$C" "$F" CFG.layer_split=26 "ARG=--prefill 4096" STRATA_MMQ_Q8K=1 STRATA_DUMP_FIRST_LOGITS=$L/k5_q8k
CFG=/tmp/basalt-prof/qwen-iq3s.json MODEL=qwen3.8-abl-iq3_s ./pab.sh i5_ctl "$C" "$F" "ARG=--prefill 4096" STRATA_DUMP_FIRST_LOGITS=$L/i5_ctl
CFG=/tmp/basalt-prof/qwen-iq3s.json MODEL=qwen3.8-abl-iq3_s ./pab.sh i5_nlperm "$N" "$F" "ARG=--prefill 4096" STRATA_DUMP_FIRST_LOGITS=$L/i5_nlperm
CFG=/tmp/basalt-prof/qwen-iq3s.json MODEL=qwen3.8-abl-iq3_s ./pab.sh i5_clamp "$K" "$F" "ARG=--prefill 4096" STRATA_DUMP_FIRST_LOGITS=$L/i5_clamp
CFG=/tmp/basalt-prof/qwen-iq3s.json MODEL=qwen3.8-abl-iq3_s ./pab.sh i5_iqq8k "$C" "$F" "ARG=--prefill 4096" STRATA_PF_IQ_Q8K=1 STRATA_DUMP_FIRST_LOGITS=$L/i5_iqq8k
} > batch24.out 2>&1
Q=/home/ronald_rump/inbox/agents/claude/strata-q38-rdna2mmq/tools/quant_quality.py
for p in "k5_ctl k5_tj16" "k5_ctl k5_q8k" "i5_ctl i5_nlperm" "i5_ctl i5_clamp" "i5_ctl i5_iqq8k"; do set -- $p; echo "== $1 vs $2"; python3 -I $Q logits $L/$1.0 $L/$2.0 --topk 20 --json $L/$1-vs-$2.json 2>&1 | tail -5; done >> batch24.out
echo WINDOW5B DONE
