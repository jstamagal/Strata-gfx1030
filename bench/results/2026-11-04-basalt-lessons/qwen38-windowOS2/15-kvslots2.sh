#!/bin/bash
# Insert: confirm STRATA_KV_PF_SLOTS=1 (card 1 -> 100% resident, zero-doorbell graph on both stages; agent +6.1% in batch36)
# at 36K prompt + agent (reverse order), and test the alternative: slots 0 with --kv-resident 61440 (does card 1 reach 11776 without the prompt-path change?).
H=/home/ronald_rump/inbox/agents/claude/strata-ab
cd $H
echo "insert kvslots2 start $(date +%H:%M)"
{
export MODEL=qwen3.8-abl-iq3_s CFG=$H/qwen-iq3s.json
env RUNS=2 MAXTOK=100 ./pab2.sh w_kvs2_36k_1 STRATA_KV_PF_SLOTS=1
env RUNS=2 MAXTOK=100 ./pab2.sh w_kvs2_36k_0 STRATA_KV_PF_SLOTS=0
./agab2.sh w_kvs2_agent_1b STRATA_KV_PF_SLOTS=1
./agab2.sh w_kvs2_agent_0b STRATA_KV_PF_SLOTS=0
./agab2.sh w_kvs2_kv61_agent STRATA_KV_PF_SLOTS=0 "ARG=--kv-resident 61440"
env RUNS=2 MAXTOK=100 ./pab2.sh w_kvs2_kv61_36k STRATA_KV_PF_SLOTS=0 "ARG=--kv-resident 61440"
for a in w_kvs2_36k_1 w_kvs2_36k_0 w_kvs2_agent_1b w_kvs2_agent_0b w_kvs2_kv61_agent w_kvs2_kv61_36k; do
  echo "== $a $(grep -o 'CUDA1 runs.*profiled pairs' ab/$a/engine.log | head -1) zero-doorbell=$(grep -c zero-doorbell ab/$a/engine.log)"; done
} > batch38.out 2>&1
echo "INSERT KVSLOTS2 DONE $(date +%H:%M)"
