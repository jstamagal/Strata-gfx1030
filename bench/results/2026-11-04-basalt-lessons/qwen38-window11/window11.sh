#!/bin/bash
# window 11 (after window 10): IQ3_S at a 120K prompt, live split 25: 2048-token chunks (live auto) vs 4096; window 8 only had 26+4096 vs 26+8192
H=/home/ronald_rump/inbox/agents/claude/strata-ab
until grep -q 'WINDOW10 DONE' $H/window10.out 2>/dev/null; do sleep 15; done
echo "window 11 start $(date +%H:%M)"
cd $H
{
export MAXTOK=100 RUNS=2 PROMPT=$H/long-prompt-110k.json MODEL=qwen3.8-abl-iq3_s
env CFG=$H/qwen-iq3s.json ./pab2.sh w11_i120k_25auto CFG.layer_split=25
env CFG=$H/qwen-iq3s.json ./pab2.sh w11_i120k_25p4k CFG.layer_split=25 "ARG=--prefill 4096"
env CFG=$H/qwen-iq3s.json ./pab2.sh w11_i120k_25auto_b CFG.layer_split=25
env CFG=$H/qwen-iq3s.json ./pab2.sh w11_i120k_25p4k_b CFG.layer_split=25 "ARG=--prefill 4096"
} > batch31.out 2>&1
echo WINDOW11 DONE
