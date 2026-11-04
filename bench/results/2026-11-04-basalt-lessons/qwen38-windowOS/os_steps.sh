#!/bin/bash
# live OS/driver steps for the OS window: os_steps.sh <step> apply|revert ; state <- prints current values
O=/home/ronald_rump/inbox/agents/claude/os-tune
lact_write() { python3 $O/lact_edit.py $1 > $O/lact-config.new.yaml && sudo -n cp $O/lact-config.new.yaml /etc/lact/config.yaml && sudo -n systemctl restart lactd && sleep 12; }
case "$1:$2" in
  lact:apply)   lact_write pin ;;
  lact:revert)  lact_write unpin ;;
  mrrs:apply)   for b in 03:00.0 07:00.0; do sudo -n setpci -s $b CAP_EXP+08.w=5000:7000; done ;;   # MaxReadReq 512 -> 4096
  mrrs:revert)  for b in 03:00.0 07:00.0; do sudo -n setpci -s $b CAP_EXP+08.w=2000:7000; done ;;
  irq:apply)    echo 30 | sudo -n tee /proc/irq/193/smp_affinity_list >/dev/null; echo 31 | sudo -n tee /proc/irq/194/smp_affinity_list >/dev/null ;;
  irq:revert)   for i in 193 194; do echo 0-31 | sudo -n tee /proc/irq/$i/smp_affinity_list >/dev/null; done ;;
  lazy:apply)   echo lazy | sudo -n tee /sys/kernel/debug/sched/preempt >/dev/null ;;
  lazy:revert)  echo full | sudo -n tee /sys/kernel/debug/sched/preempt >/dev/null ;;
  aspm:apply)   for b in 07:00.0 06:00.0 05:00.0 00:01.3; do sudo -n setpci -s $b CAP_EXP+10.w=0:3; done ;;   # card1 route: ASPM L1 -> off (endpoint first)
  aspm:revert)  for b in 00:01.3 05:00.0 06:00.0 07:00.0; do sudo -n setpci -s $b CAP_EXP+10.w=2:3; done ;;   # back to L1 as the BIOS set it
  cap275:apply)  python3 $O/lact_cap.py 275 > $O/lact-config.new.yaml && sudo -n cp $O/lact-config.new.yaml /etc/lact/config.yaml && sudo -n systemctl restart lactd && sleep 12 ;;
  cap275:revert) python3 $O/lact_cap.py 260 > $O/lact-config.new.yaml && sudo -n cp $O/lact-config.new.yaml /etc/lact/config.yaml && sudo -n systemctl restart lactd && sleep 12 ;;
  state:*)
    for c in card0 card1; do d=/sys/class/drm/$c/device; h=$(ls -d $d/hwmon/hwmon*|head -1); echo "$c cap=$(( $(cat $h/power1_cap)/1000000 ))W mclk=[$(tr '\n' ' ' < $d/pp_dpm_mclk)] od_mclk=$(grep -A2 OD_MCLK $d/pp_od_clk_voltage | tail -1 | tr -d ' ') sclk_od=$(grep -A2 OD_SCLK $d/pp_od_clk_voltage | tail -1 | tr -d ' ')"; done
    for b in 03:00.0 07:00.0; do echo "$b DevCtl=$(sudo -n setpci -s $b CAP_EXP+08.w)"; done
    for b in 00:01.3 05:00.0 06:00.0 07:00.0; do printf "%s %s; " $b "$(sudo -n lspci -vv -s $b | grep -oE 'LnkCtl:.*ASPM [^;]+' | grep -oE 'ASPM [^;]+')"; done; echo
    echo "irq193->$(cat /proc/irq/193/effective_affinity_list) irq194->$(cat /proc/irq/194/effective_affinity_list) preempt=$(sudo -n cat /sys/kernel/debug/sched/preempt)" ;;
  *) echo "unknown $1 $2"; exit 2 ;;
esac
