#!/bin/bash
# window 5 (starts 10:15 when no builds run): prompt split x prefill sweeps on all three packs, reservation fix check, IQ3_S proxy
until [ "$(date +%H%M)" -ge 1015 ]; do sleep 20; done
quiet=0; while [ $quiet -lt 3 ]; do if pgrep -f '[c]make --build|[n]inja|[h]ipcc|[l]d.lld|[c]lang.*-cc1' >/dev/null; then quiet=0; else quiet=$((quiet+1)); fi; sleep 10; done
echo "window 5 start $(date +%H:%M)"
cd /tmp/basalt-prof
B=EXE=/home/ronald_rump/Strata_donttouch/build-hip/strata
{
export MAXTOK=300 RUNS=2
for v in "26 4096" "26 2048" "26 8192" "25 4096" "27 4096" "28 8192"; do set -- $v; CFG=/tmp/basalt-prof/qwen-ud4.json ./pab.sh y$1p$2 "$B" CFG.layer_split=$1 "ARG=--prefill $2"; done
CFG=/tmp/basalt-prof/qwen-iq3s.json MODEL=qwen3.8-abl-iq3_s ./pab.sh iq25 "$B"
for s in 26 27 28; do CFG=/tmp/basalt-prof/qwen-iq3s.json MODEL=qwen3.8-abl-iq3_s ./pab.sh iq${s}p4k "$B" CFG.layer_split=$s "ARG=--prefill 4096"; done
CFG=/tmp/basalt-prof/qwen-q40.json MODEL=swift-qwen3.8-bartowski-q4_0 ./pab.sh q426 "$B"
for s in 26 27 28; do CFG=/tmp/basalt-prof/qwen-q40.json MODEL=swift-qwen3.8-bartowski-q4_0 ./pab.sh q4${s}p4k "$B" CFG.layer_split=$s "ARG=--prefill 4096"; done
} > batch17.out 2>&1
{
F="ARG=--adapt-swaps 0"
CFG=/tmp/basalt-prof/qwen-ud4.json RUNS=2 ./ab.sh r_fix EXE=/home/ronald_rump/inbox/agents/claude/strata-q38-int/build/strata-int-225819880 "$F" CFG.layer_split=24
CFG=/tmp/basalt-prof/qwen-iq3s.json MODEL=qwen3.8-abl-iq3_s RUNS=2 ./ab.sh i3_base "$B" "$F"
CFG=/tmp/basalt-prof/qwen-iq3s.json MODEL=qwen3.8-abl-iq3_s RUNS=2 ./ab.sh i3_proxy "$B" "$F" STRATA_MTP_HEAD_PROXY=1
} > batch18.out 2>&1
echo WINDOW5 DONE
