#!/bin/bash
# window OS (after window 12, before 13): live OS/driver fixes as a cumulative ladder on IQ3_S with auto-revert.
# Steps: lact = LACT PF1 memory power state pinned to the top level [3] (no clock raised; also fixes LACT's 150 W fallback);
# mrrs = PCIe MaxReadReq 512 -> 4096 on both V620s; irq = the two cards' interrupts off cpu5/6 (pool cores) onto 30/31;
# lazy = kernel preemption full -> lazy. Keep a step if agent tok/s >= best - 0.3% and prompt >= best - 1.5%, else revert it.
H=/home/ronald_rump/inbox/agents/claude/strata-ab; O=/home/ronald_rump/inbox/agents/claude/os-tune
until grep -q 'WINDOW12 DONE' $H/window12.out 2>/dev/null; do sleep 15; done
echo "window OS start $(date +%H:%M)"
cd $H
I="CFG=$H/qwen-iq3s.json MODEL=qwen3.8-abl-iq3_s"; LOG=$H/batchOS.out
measure() { env $I RUNS=2 ./pab2.sh $1_prompt >> $LOG 2>&1; env $I ./agab2.sh $1_agent >> $LOG 2>&1; }
agent() { grep "^$1_agent TOTAL" $LOG | tail -1 | awk '{print $5}'; }
prompt() { grep "^$1_prompt run" $LOG | awk '{s+=$5; n++} END {if (n) printf "%.1f", s/n; else print 0}'; }
{ echo "== state before"; $O/os_steps.sh state x; } >> $LOG 2>&1
measure w_os0; BA=$(agent w_os0); BP=$(prompt w_os0); echo "baseline agent $BA prompt $BP" >> $LOG
for s in lact mrrs irq lazy; do
  $O/os_steps.sh $s apply >> $LOG 2>&1; { echo "== state after $s apply"; $O/os_steps.sh state x; } >> $LOG 2>&1
  measure w_os_$s; A=$(agent w_os_$s); P=$(prompt w_os_$s)
  if python3 -c "import sys; a,p,ba,bp=map(float,sys.argv[1:]); sys.exit(0 if a>=ba*0.997 and p>=bp*0.985 else 1)" "${A:-0}" "${P:-0}" "$BA" "$BP"; then
    echo "KEEP $s: agent $A (best $BA) prompt $P (best $BP)" >> $LOG
    BA=$(python3 -c "print(max($BA,${A:-0}))"); BP=$(python3 -c "print(max($BP,${P:-0}))")
  else
    echo "REVERT $s: agent $A (best $BA) prompt $P (best $BP)" >> $LOG; $O/os_steps.sh $s revert >> $LOG 2>&1
  fi
done
{ echo "== final state"; $O/os_steps.sh state x; } >> $LOG 2>&1
echo WINDOWOS DONE
