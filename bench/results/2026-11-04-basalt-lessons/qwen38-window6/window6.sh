#!/bin/bash
# window 6 (after window 5): TP-prompt route-ahead + layer-major residual pairs, RDNA2 experts zero-head decode, IQ3_S Q4_0 draft head, DF2 serial on agent workloads
until grep -q 'WINDOW5B DONE' /tmp/basalt-prof/window5b.out 2>/dev/null; do sleep 15; done
echo "window 6 start $(date +%H:%M)"
cd /tmp/basalt-prof
T=/home/ronald_rump/inbox/agents/claude/strata-q38-tpprompt/bench/results/2026-11-04-basalt-lessons/qwen38-tpprompt
{
export MAXTOK=300 RUNS=2
for c in pf2048-control pf2048-ahead16 pf4096-control pf4096-ahead16 pf4096-ahead32 lm4096-host-later lm4096-vram-later; do CFG=$T/$c.json ./pab.sh t_$c; done
} > batch19.out 2>&1
{
F="ARG=--adapt-swaps 0"
R=/home/ronald_rump/inbox/agents/claude/strata-q38-rdna2mmq/build
CFG=/tmp/basalt-prof/qwen-ud4.json RUNS=2 ./ab.sh k_control EXE=/tmp/basalt-prof/strata-rdna2-control "$F" CFG.layer_split=24
CFG=/tmp/basalt-prof/qwen-ud4.json RUNS=2 ./ab.sh k_expdotz EXE=/tmp/basalt-prof/strata-rdna2-experts-dotz "$F" CFG.layer_split=24
CFG=/tmp/basalt-prof/qwen-iq3s.json MODEL=qwen3.8-abl-iq3_s RUNS=2 ./ab.sh i3_q4head "$F" "ARG=--mtp-q4 head"
} > batch20.out 2>&1
{
X=EXE=/home/ronald_rump/inbox/agents/claude/strata-q38-dflash2/build/strata-ab
CFG=/tmp/basalt-prof/qwen-ud4.json ./agab.sh g_mtp_pw "$X" CFG.layer_split=24
CFG=/tmp/basalt-prof/qwen-dflash2-nopw-baseline.json ./agab.sh g_df2_s6c035 "$X"
CFG=/tmp/basalt-prof/qwen-dflash2-s5-c050-baseline.json ./agab.sh g_df2_s5c050 "$X"
CFG=/tmp/basalt-prof/qwen-dflash2-s4-c050-baseline.json ./agab.sh g_df2_s4c050 "$X"
CFG=/tmp/basalt-prof/qwen-mtp-nopw-baseline.json ./agab.sh g_mtp_nopw "$X"
} > batch21.out 2>&1
echo WINDOW6 DONE
