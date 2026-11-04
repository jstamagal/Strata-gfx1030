#!/bin/bash
# window 9 (after window 8): pipelined DFlash2 at full block widths (spec 7/8) and lower/no confidence cutoff, agent workload
H=/home/ronald_rump/inbox/agents/claude/strata-ab
until grep -q 'WINDOW8 DONE' $H/window8.out 2>/dev/null; do sleep 15; done
echo "window 9 start $(date +%H:%M)"
cd $H
X=EXE=/home/ronald_rump/inbox/agents/claude/strata-q38-dflash2/build/strata-ab
C=$H/qwen-dflash2-pw.json
# whole-window acceptance needs the pipeline trace (DF2 agent); same overhead on every arm incl. the MTP control
T="STRATA_PIPELINE_DEBUG=1 STRATA_PIPELINE_TRACE=pipeline.trace"
{
env CFG=$C ./agab2.sh w9_df2_s7c035 "$X" $T "ARG=--spec 7" "ARG=--draft-conf 0.35"
env CFG=$C ./agab2.sh w9_df2_s8c035 "$X" $T "ARG=--spec 8" "ARG=--draft-conf 0.35"
env CFG=$C ./agab2.sh w9_df2_s8c020 "$X" $T "ARG=--spec 8" "ARG=--draft-conf 0.2"
env CFG=$C ./agab2.sh w9_df2_s8c001 "$X" $T "ARG=--spec 8" "ARG=--draft-conf 0.01"
env CFG=$H/qwen-ud4.json ./agab2.sh w9_mtp_pw "$X" CFG.layer_split=24 $T
} > batch29.out 2>&1
echo WINDOW9 DONE
