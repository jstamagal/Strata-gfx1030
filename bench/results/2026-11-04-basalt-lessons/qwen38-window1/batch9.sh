#!/bin/bash
# rocBLAS table for the running build (5.8.0.565c4c74), ~36K prompt, main exe, live config
export CFG=/tmp/basalt-prof/qwen-ud4.json MODEL=qwen3.8-flash-next-unsloth-ud-q4_k_xl
cd /tmp/basalt-prof
RUNS=2 ./pab.sh r_base
RUNS=2 ./pab.sh r_new STRATA_ROCBLAS_TUNING=/home/ronald_rump/inbox/agents/claude/strata-q38-gdnpf/tools/hip
RUNS=2 ./pab.sh r_base2
RUNS=2 ./pab.sh r_new2 STRATA_ROCBLAS_TUNING=/home/ronald_rump/inbox/agents/claude/strata-q38-gdnpf/tools/hip
echo BATCH DONE
