#!/bin/bash
# window 13 (after the OS window): CPU layout across the 9950X's two CCDs (CCD0 = cores 0-7 + SMT 16-23, CCD1 = 8-15 + 24-31), IQ3_S.
# Live today: host thread on cpu0, 4 workers on 1-4 (all CCD0). Window 12's pw8 = workers 1-8 (straddles), pw0 = 1-15 (both).
H=/home/ronald_rump/inbox/agents/claude/strata-ab
until grep -q 'WINDOWOS DONE' $H/windowos.out 2>/dev/null; do sleep 15; done
echo "window 13 start $(date +%H:%M)"
cd $H
I="CFG=$H/qwen-iq3s.json MODEL=qwen3.8-abl-iq3_s"
run() { n=$1; shift; env $I RUNS=2 ./pab2.sh ${n}_prompt "$@"; env $I ./agab2.sh ${n}_agent "$@"; }
{
run w13_c0pool7  "ARG=--pool-cores 1-7"                                   # host + 7 workers on CCD0 (helpers float)
run w13_c1pool8  "ARG=--pool-cores 8-15"                                  # host CCD0, 8 workers on CCD1
run w13_c1all7   "ARG=--host-core last" "ARG=--pool-cores 8-14"           # host cpu15 + 7 workers, all CCD1
env WRAP="taskset -c 0-7,16-23" $I RUNS=2 ./pab2.sh w13_c0whole_prompt "ARG=--pool-cores 1-7"   # whole process incl. io/stage/adapt/vision on CCD0
env WRAP="taskset -c 0-7,16-23" $I ./agab2.sh w13_c0whole_agent "ARG=--pool-cores 1-7"
} > batch33.out 2>&1
echo WINDOW13 DONE
