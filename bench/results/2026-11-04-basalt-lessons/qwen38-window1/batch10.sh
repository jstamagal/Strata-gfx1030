#!/bin/bash
# head agent: proxy auto-on with env unset, then + Q8 verify screen; deterministic bench, md5 must stay d5dcd30186c2
export CFG=/tmp/basalt-prof/qwen-ud4.json MODEL=qwen3.8-flash-next-unsloth-ud-q4_k_xl RUNS=3
cd /tmp/basalt-prof
H=EXE=/home/ronald_rump/inbox/agents/claude/strata-q38-head/build/strata
F="ARG=--adapt-swaps 0"
./ab.sh a_auto "$H" "$F"
./ab.sh a_q8 "$H" "$F" STRATA_HEAD_Q8_SCREEN=1
./ab.sh a_off "$H" "$F" STRATA_MTP_HEAD_PROXY=0
grep -a -i "2-bit head\|reserv" /tmp/basalt-prof/ab/a_auto/engine.log | head -5
echo BATCH DONE
