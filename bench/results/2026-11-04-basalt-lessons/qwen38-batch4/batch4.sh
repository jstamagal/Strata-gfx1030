#!/bin/bash
# confirm STRATA_MTP_HEAD_PROXY=1 on the live (adaptive) config, head exe both sides, 2 loads per side, 4 essays per load
export CFG=/tmp/basalt-prof/qwen-ud4.json MODEL=qwen3.8-flash-next-unsloth-ud-q4_k_xl RUNS=4
cd /tmp/basalt-prof
H=EXE=/home/ronald_rump/inbox/agents/claude/strata-q38-head/build/strata
Z="STRATA_HEAD_Q8_SCREEN=0 STRATA_HEAD_Q6_SCREEN=0"
./ab.sh l_base1 "$H" STRATA_MTP_HEAD_PROXY=0 $Z
./ab.sh l_proxy1 "$H" STRATA_MTP_HEAD_PROXY=1 $Z
./ab.sh l_base2 "$H" STRATA_MTP_HEAD_PROXY=0 $Z
./ab.sh l_proxy2 "$H" STRATA_MTP_HEAD_PROXY=1 $Z
echo BATCH DONE
