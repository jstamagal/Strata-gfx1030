#!/bin/bash
# OS window 2: ASPM off on card 1's route (permanent by KING's ruling; boot unit gpu-pcie-tune.service persists it), then env-only arms:
# HSA_ENABLE_INTERRUPT=0 (polling completion), HSA_ENABLE_MWAITX=0 (busy-wait without MWAITX). Starts after the lookup16 window.
H=/home/ronald_rump/inbox/agents/claude/strata-ab; O=/home/ronald_rump/inbox/agents/claude/os-tune
until grep -q 'WINDOWL16 DONE' $H/windowl16.out 2>/dev/null; do sleep 15; done
source $H/os2p-hook.sh; os2p_hook
echo "OS window 2 start $(date +%H:%M)"
cd $H
I="CFG=$H/qwen-iq3s.json MODEL=qwen3.8-abl-iq3_s"; LOG=$H/batchOS2.out
measure() { n=$1; shift; env $I RUNS=2 ./pab2.sh ${n}_prompt "$@" >> $LOG 2>&1; env $I ./agab2.sh ${n}_agent "$@" >> $LOG 2>&1; }
agent() { grep "^$1_agent TOTAL" $LOG | tail -1 | awk '{print $5}'; }
prompt() { grep "^$1_prompt run" $LOG | awk '{s+=$5; n++} END {if (n) printf "%.1f", s/n; else print 0}'; }
{ echo "== state before"; $O/os_steps.sh state x; } >> $LOG 2>&1
measure w_os2_0; BA=$(agent w_os2_0); BP=$(prompt w_os2_0); echo "baseline agent $BA prompt $BP" >> $LOG
$O/os_steps.sh aspm apply >> $LOG 2>&1; { echo "== state after aspm apply (permanent)"; $O/os_steps.sh state x; } >> $LOG 2>&1
measure w_os2_aspm; A=$(agent w_os2_aspm); P=$(prompt w_os2_aspm); echo "ASPM-OFF (kept by ruling): agent $A prompt $P (baseline $BA / $BP)" >> $LOG
BA=$(python3 -c "print(max($BA,${A:-0}))"); BP=$(python3 -c "print(max($BP,${P:-0}))")
measure w_os2_poll HSA_ENABLE_INTERRUPT=0; echo "ENV poll: agent $(agent w_os2_poll) prompt $(prompt w_os2_poll) (best $BA / $BP)" >> $LOG
measure w_os2_nomwaitx HSA_ENABLE_MWAITX=0; echo "ENV mwaitx-off: agent $(agent w_os2_nomwaitx) prompt $(prompt w_os2_nomwaitx) (best $BA / $BP)" >> $LOG
{ echo "== final state"; $O/os_steps.sh state x; } >> $LOG 2>&1
echo WINDOWOS2A DONE
