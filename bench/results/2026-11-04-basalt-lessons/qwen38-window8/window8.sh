#!/bin/bash
# window 8 (after window 7): IQ3_S split/chunk points, IQ3_S STRATA_IQ_GATHER decode, ~120K-token prompt 4096 vs 8192 chunks
until grep -q 'WINDOW7 DONE' /home/ronald_rump/inbox/agents/claude/strata-ab/window7.out 2>/dev/null; do sleep 15; done
echo "window 8 start $(date +%H:%M)"
H=/home/ronald_rump/inbox/agents/claude/strata-ab; cd $H
I3="CFG=$H/qwen-iq3s.json"; export MODEL=qwen3.8-abl-iq3_s
{
export MAXTOK=300 RUNS=2
env CFG=$H/qwen-iq3s.json ./pab2.sh w8_iq25 CFG.layer_split=25
env CFG=$H/qwen-iq3s.json ./pab2.sh w8_iq25p4k CFG.layer_split=25 "ARG=--prefill 4096"
env CFG=$H/qwen-iq3s.json ./pab2.sh w8_iq24p4k CFG.layer_split=24 "ARG=--prefill 4096"
env CFG=$H/qwen-iq3s.json ./pab2.sh w8_iq26p4k CFG.layer_split=26 "ARG=--prefill 4096"
} > batch26.out 2>&1
{
F="ARG=--adapt-swaps 0"
env CFG=$H/qwen-iq3s.json RUNS=2 ./ab2.sh w8_i3_base "$F"
env CFG=$H/qwen-iq3s.json RUNS=2 ./ab2.sh w8_i3_gather "$F" STRATA_IQ_GATHER=1
} > batch27.out 2>&1
{
export MAXTOK=100 RUNS=2 PROMPT=$H/long-prompt-110k.json
env CFG=$H/qwen-ud4-live.json MODEL=qwen3.8-flash-next-unsloth-ud-q4_k_xl ./pab2.sh w8_u120k_p4k
env CFG=$H/qwen-ud4-live.json MODEL=qwen3.8-flash-next-unsloth-ud-q4_k_xl ./pab2.sh w8_u120k_p8k "ARG=--prefill 8192"
env CFG=$H/qwen-iq3s.json ./pab2.sh w8_i120k_p4k CFG.layer_split=26 "ARG=--prefill 4096"
env CFG=$H/qwen-iq3s.json ./pab2.sh w8_i120k_p8k CFG.layer_split=26 "ARG=--prefill 8192"
} > batch28.out 2>&1
echo WINDOW8 DONE
