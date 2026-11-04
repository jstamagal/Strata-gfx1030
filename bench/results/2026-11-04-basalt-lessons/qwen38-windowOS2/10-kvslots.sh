#!/bin/bash
# Insert: STRATA_KV_PF_SLOTS=1 (bounded prompt-KV staging; drops the full-context staging pool, ~528 MiB/card at 512K, and
# re-prices the owned prompt reserve) vs 0, IQ3_S live config. 120K prompt ABBA (prompt + 100-token gen) and agent decode.
H=/home/ronald_rump/inbox/agents/claude/strata-ab
cd $H
echo "insert kvslots start $(date +%H:%M)"
{
export MODEL=qwen3.8-abl-iq3_s CFG=$H/qwen-iq3s.json
env MAXTOK=100 RUNS=2 PROMPT=$H/long-prompt-110k.json ./pab2.sh w_kvs_120k_0 STRATA_KV_PF_SLOTS=0
env MAXTOK=100 RUNS=2 PROMPT=$H/long-prompt-110k.json ./pab2.sh w_kvs_120k_1 STRATA_KV_PF_SLOTS=1
env MAXTOK=100 RUNS=2 PROMPT=$H/long-prompt-110k.json ./pab2.sh w_kvs_120k_1b STRATA_KV_PF_SLOTS=1
env MAXTOK=100 RUNS=2 PROMPT=$H/long-prompt-110k.json ./pab2.sh w_kvs_120k_0b STRATA_KV_PF_SLOTS=0
./agab2.sh w_kvs_agent_0 STRATA_KV_PF_SLOTS=0
./agab2.sh w_kvs_agent_1 STRATA_KV_PF_SLOTS=1
for a in w_kvs_120k_0 w_kvs_120k_1 w_kvs_agent_0 w_kvs_agent_1; do echo "== $a cache/staging lines"; grep -iE 'expert cache|slots|staging|kv_pf|reserve' ab/$a/engine.log | head -8; done
} > batch36.out 2>&1
echo "INSERT KVSLOTS DONE $(date +%H:%M)"
