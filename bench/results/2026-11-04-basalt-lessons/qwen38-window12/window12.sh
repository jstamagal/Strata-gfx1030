#!/bin/bash
# window 12 (after window 11): --pool-workers sweep on IQ3_S (4 was a hunch). 0 = engine default (physical cores minus the host's = 15).
# Each count: 36K prompt x2 (prompt speed; CPU share of prompt experts) and the 6-case agent workload (decode).
H=/home/ronald_rump/inbox/agents/claude/strata-ab
until grep -q 'WINDOW11 DONE' $H/window11.out 2>/dev/null; do sleep 15; done
echo "window 12 start $(date +%H:%M)"
cd $H
{
for W in 4 8 12 0; do
  env CFG=$H/qwen-iq3s.json MODEL=qwen3.8-abl-iq3_s RUNS=2 ./pab2.sh w12_pw${W}_prompt "ARG=--pool-workers $W"
  env CFG=$H/qwen-iq3s.json MODEL=qwen3.8-abl-iq3_s ./agab2.sh w12_pw${W}_agent "ARG=--pool-workers $W"
done
} > batch32.out 2>&1
echo WINDOW12 DONE
