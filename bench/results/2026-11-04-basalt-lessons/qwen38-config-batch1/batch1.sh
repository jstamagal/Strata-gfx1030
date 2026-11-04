#!/bin/bash
# Qwen3.8 UD-Q4_K_XL config-only A/Bs from the discovery agent (essay x2, greedy, thinking off)
export CFG=/tmp/basalt-prof/qwen-ud4.json MODEL=qwen3.8-flash-next-unsloth-ud-q4_k_xl RUNS=2
cd /tmp/basalt-prof
# wait for agents' builds to drain so RAM is free for the 77 GB arena
for i in $(seq 1 120); do pgrep -x ninja >/dev/null || break; sleep 10; done
echo "builds drained after $((i*10)) s; $(free -g | awk '/Mem:/{print $7}') GB available"
./ab.sh q_base
./ab.sh q_stage0 STRATA_MTP_STAGE0=1
./ab.sh q_kqsplit STRATA_KQ_SPLIT=1
./ab.sh q_v2k STRATA_EXPERT_V2K=1
./ab.sh q_theta05 STRATA_PIPELINE_THETA=0.5
./ab.sh q_mtpt4 "ARG=--mtp-max-t 4"
./ab.sh q_adapt4 "ARG=--adapt-every 4"
./ab.sh q_base2
echo BATCH DONE
