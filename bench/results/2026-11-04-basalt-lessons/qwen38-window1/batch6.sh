#!/bin/bash
# GDN-agent switches, one at a time, deterministic bench (--adapt-swaps 0), md5 must stay d5dcd30186c2
export CFG=/tmp/basalt-prof/qwen-ud4.json MODEL=qwen3.8-flash-next-unsloth-ud-q4_k_xl RUNS=2
cd /tmp/basalt-prof
G=EXE=/home/ronald_rump/inbox/agents/claude/strata-q38-gdn/build/strata
F="ARG=--adapt-swaps 0"
./ab.sh g_base "$G" "$F"
./ab.sh g_lazy "$G" "$F" STRATA_GDN_LAZY_COMMIT=1
./ab.sh g_vfuse "$G" "$F" STRATA_VERIFY_INPUT_FUSE=1
./ab.sh g_mtpkv "$G" "$F" STRATA_MTP_KV_BRANCH=1
./ab.sh g_qsaqf "$G" "$F" STRATA_QSA_QUERY_FIRST=1 STRATA_QSA_GATE_BLOCKS=0
./ab.sh g_ple "$G" "$F" STRATA_PLE_SIDE_BRANCH=1
./ab.sh g_hpref "$G" "$F" STRATA_MTP_HEAD_PREFETCH=1
./ab.sh g_base2 "$G" "$F"
echo BATCH DONE
