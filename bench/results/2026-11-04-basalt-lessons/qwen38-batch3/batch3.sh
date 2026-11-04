#!/bin/bash
# window 3: head-agent MTP head proxy (deterministic bench) + prompt side-stream A/B
export CFG=/tmp/basalt-prof/qwen-ud4.json MODEL=qwen3.8-flash-next-unsloth-ud-q4_k_xl
cd /tmp/basalt-prof
H=EXE=/home/ronald_rump/inbox/agents/claude/strata-q38-head/build/strata
F="ARG=--adapt-swaps 0"
RUNS=3 ./ab.sh h_base "$H" "$F" STRATA_MTP_HEAD_PROXY=0 STRATA_HEAD_Q8_SCREEN=0 STRATA_HEAD_Q6_SCREEN=0
RUNS=3 ./ab.sh h_proxy "$H" "$F" STRATA_MTP_HEAD_PROXY=1 STRATA_HEAD_Q8_SCREEN=0 STRATA_HEAD_Q6_SCREEN=0
RUNS=2 ./pab.sh p_side1
RUNS=2 ./pab.sh p_side0 STRATA_PF_SIDE_STREAM=0
echo BATCH DONE
