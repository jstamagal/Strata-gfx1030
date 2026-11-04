#!/bin/bash
# integration exe (main + basalt38/head + basalt38/tier): frozen check, then live adaptive tier switches
export CFG=/tmp/basalt-prof/qwen-ud4.json MODEL=qwen3.8-flash-next-unsloth-ud-q4_k_xl
cd /tmp/basalt-prof
I=EXE=/home/ronald_rump/inbox/agents/claude/strata-q38-int/build/strata
F="ARG=--adapt-swaps 0"
{
RUNS=2 ./ab.sh i_base "$I" "$F"
RUNS=2 ./ab.sh i_ra "$I" "$F" STRATA_STALL_READAHEAD=1
} > batch11.out 2>&1
{
for pass in 1 2; do
  RUNS=4 ./ab.sh i_lbase$pass "$I"
  RUNS=4 ./ab.sh i_lra$pass "$I" STRATA_STALL_READAHEAD=1
  RUNS=4 ./ab.sh i_learly$pass "$I" STRATA_ADAPT_EARLY_RANK=1
  RUNS=4 ./ab.sh i_lidle$pass "$I" STRATA_ADAPT_IDLE_GAPS=1
  RUNS=4 ./ab.sh i_lacross$pass "$I" STRATA_ADAPT_ACROSS=1
  RUNS=4 ./ab.sh i_lonce$pass "$I" STRATA_ADAPT_ONCE=1
done
} > batch12.out 2>&1
echo WINDOW DONE
