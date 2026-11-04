#!/bin/bash
# kernel-agent ports on the deterministic bench (--adapt-swaps 0); greedy md5 must stay d5dcd30186c2
export CFG=/tmp/basalt-prof/qwen-ud4.json MODEL=qwen3.8-flash-next-unsloth-ud-q4_k_xl
cd /tmp/basalt-prof
K=EXE=/home/ronald_rump/inbox/agents/claude/strata-q38-kern/build/strata
F="ARG=--adapt-swaps 0"
RUNS=2 ./ab.sh k_base "$K" "$F"
RUNS=2 ./ab.sh k_route "$K" "$F" STRATA_ROUTE_MASK=1
RUNS=2 ./ab.sh k_qsa "$K" "$F" STRATA_QSA_ROWS4=1
RUNS=2 ./ab.sh k_topk "$K" "$F" STRATA_TOPK_SUFFIX=1
RUNS=2 ./ab.sh k_embed "$K" "$F" STRATA_EMBED_HOT=1
# sampled (temp 0.7, top-k 20, top-p 0.95, fixed seed) for the one-launch sampler
S='"temperature":0.7,"top_k":20,"top_p":0.95,"seed":1234'
SAMP="$S" RUNS=2 ./ab.sh k_samp_base "$K" "$F"
SAMP="$S" RUNS=2 ./ab.sh k_samp_once "$K" "$F" STRATA_SAMPLER_ONCE=1
# long context decode (~36K prompt + 400 generated) for the 4-row QSA
MAXTOK=400 RUNS=1 ./pab.sh k_long_base "$K" "$F"
MAXTOK=400 RUNS=1 ./pab.sh k_long_qsa "$K" "$F" STRATA_QSA_ROWS4=1
MAXTOK=400 RUNS=1 ./pab.sh k_long_topk "$K" "$F" STRATA_TOPK_SUFFIX=1
echo BATCH DONE
