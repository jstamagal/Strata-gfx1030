#!/bin/bash
# window 7 (after window 6): pipelined DFlash2 qualification (greedy identity frozen, forced misses) then speed vs MTP+PW
until grep -q 'WINDOW6 DONE' /tmp/basalt-prof/window6.out 2>/dev/null; do sleep 15; done
echo "window 7 start $(date +%H:%M)"
cd /tmp/basalt-prof
X=EXE=/home/ronald_rump/inbox/agents/claude/strata-q38-dflash2/build/strata-ab
F="ARG=--adapt-swaps 0"
{
CFG=/tmp/basalt-prof/qwen-dflash2-nopw.json RUNS=2 ./ab.sh f_df2_nopw "$X" "$F"
CFG=/tmp/basalt-prof/qwen-dflash2-pw.json RUNS=2 ./ab.sh f_df2_pw "$X" "$F"
CFG=/tmp/basalt-prof/qwen-dflash2-pw.json RUNS=2 ./ab.sh f_df2_pw_miss "$X" "$F" STRATA_PIPELINE_DEBUG=1 STRATA_PIPELINE_FORCE_MISS=2 STRATA_PIPELINE_THETA=0
CFG=/tmp/basalt-prof/qwen-ud4.json RUNS=2 ./ab.sh f_mtp_pw "$X" "$F" CFG.layer_split=24
} > batch22.out 2>&1
{
CFG=/tmp/basalt-prof/qwen-dflash2-pw.json ./agab.sh g_df2_pw "$X"
CFG=/tmp/basalt-prof/qwen-dflash2-pw.json RUNS=4 ./ab.sh e_df2_pw "$X"
CFG=/tmp/basalt-prof/qwen-ud4.json RUNS=4 ./ab.sh e_mtp_pw "$X" CFG.layer_split=24
} > batch23.out 2>&1
{
export MAXTOK=300 RUNS=2
CFG=/tmp/basalt-prof/qwen-dflash2-pw.json ./pab.sh p_df2_pw "$X"
CFG=/tmp/basalt-prof/qwen-ud4.json ./pab.sh p_mtp_pw "$X" CFG.layer_split=24
} > batch25.out 2>&1
echo WINDOW7 DONE
