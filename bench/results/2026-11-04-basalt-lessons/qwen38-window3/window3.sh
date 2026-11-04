#!/bin/bash
# window 3: reservation-fix check, Q8-screen exact-fallback control, layer split toward the x8 card (decode frozen + prompt)
export CFG=/tmp/basalt-prof/qwen-ud4.json MODEL=qwen3.8-flash-next-unsloth-ud-q4_k_xl
cd /tmp/basalt-prof
INT=/home/ronald_rump/inbox/agents/claude/strata-q38-int/build
HEAD=EXE=/home/ronald_rump/inbox/agents/claude/strata-q38-head/build/strata-ab-3f5f5450
F="ARG=--adapt-swaps 0"
PEN='"temperature":0,"frequency_penalty":1e-20,"penalty_last_n":64'
{
RUNS=2 ./ab.sh w3_old EXE=$INT/strata-int-11db64ca4 "$F"
RUNS=2 ./ab.sh w3_new EXE=$INT/strata "$F"
RUNS=2 SAMP="$PEN" ./ab.sh q_ctl "$HEAD" "$F"
RUNS=2 SAMP="$PEN" ./ab.sh q_q8 "$HEAD" "$F" STRATA_HEAD_Q8_SCREEN=1
for s in 26 28 32; do RUNS=2 ./ab.sh s$s EXE=$INT/strata "$F" CFG.layer_split=$s; done
} > batch13.out 2>&1
{
for s in 24 28 32; do RUNS=2 ./pab.sh p$s EXE=$INT/strata CFG.layer_split=$s; done
} > batch14.out 2>&1
echo WINDOW DONE
