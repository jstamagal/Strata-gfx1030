#!/bin/bash
# Qwen3.8 UD-Q4_K_XL kernel switches with expert placement frozen (--adapt-swaps 0), essay x3, greedy
export CFG=/tmp/basalt-prof/qwen-ud4.json MODEL=qwen3.8-flash-next-unsloth-ud-q4_k_xl RUNS=3
cd /tmp/basalt-prof
F="ARG=--adapt-swaps 0"
./ab.sh f_base "$F"
./ab.sh f_kqsplit "$F" STRATA_KQ_SPLIT=1
./ab.sh f_v2k "$F" STRATA_EXPERT_V2K=1
./ab.sh f_base2 "$F"
echo BATCH DONE
