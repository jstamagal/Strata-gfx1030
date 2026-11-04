#!/bin/bash
# window 4: prompt-first layer split sweep on main code (int-11db64ca4 exe), live config, 36K prompt + 300 tokens of long-context decode
until grep -q 'WINDOW3B DONE' /tmp/basalt-prof/window3b.out 2>/dev/null; do sleep 10; done
export CFG=/tmp/basalt-prof/qwen-ud4.json MODEL=qwen3.8-flash-next-unsloth-ud-q4_k_xl MAXTOK=300 RUNS=2
cd /tmp/basalt-prof
E=EXE=/home/ronald_rump/inbox/agents/claude/strata-q38-int/build/strata-int-11db64ca4
{
for s in 24 26 27 28 29 30; do ./pab.sh x$s "$E" CFG.layer_split=$s; done
./pab.sh x28p4k "$E" CFG.layer_split=28 "ARG=--prefill 4096"
} > batch16.out 2>&1
echo WINDOW4 DONE
