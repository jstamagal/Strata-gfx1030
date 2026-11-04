#!/bin/bash
# tier-agent draft switches on the deterministic bench (--adapt-swaps 0): output must stay md5 d5dcd30186c2
export CFG=/tmp/basalt-prof/qwen-ud4.json MODEL=qwen3.8-flash-next-unsloth-ud-q4_k_xl RUNS=2
cd /tmp/basalt-prof
T=EXE=/home/ronald_rump/inbox/agents/claude/strata-q38-tier/build/strata-basalt38-ab
F="ARG=--adapt-swaps 0"
./ab.sh d_base "$T" "$F"
./ab.sh d_cum "$T" "$F" STRATA_DRAFT_CUM=1 STRATA_DRAFT_CUM_TRACE=1
./ab.sh d_kvside "$T" "$F" STRATA_MTP_KV_SIDE=1
./ab.sh d_ctx "$T" "$F" STRATA_DRAFT_CONTEXT=1
./ab.sh d_readahead "$T" "$F" STRATA_STALL_READAHEAD=1
./ab.sh d_cpuwarm "$T" "$F" STRATA_CPU_EXPERT_WARM=1 STRATA_CPU_EXPERT_WARM_TRACE=1
echo BATCH DONE
