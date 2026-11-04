#!/bin/bash
# window 3b: serial DFlash2 vs MTP on build-hip (main), live adaptive configs, after window 3
until grep -q 'WINDOW DONE' /tmp/basalt-prof/window3.out 2>/dev/null; do sleep 10; done
cd /tmp/basalt-prof
export MODEL=qwen3.8-flash-next-unsloth-ud-q4_k_xl
{
CFG=/tmp/basalt-prof/qwen-ud4.json RUNS=4 ./ab.sh d_mtp_pw
CFG=/tmp/basalt-prof/qwen-mtp-nopw-baseline.json RUNS=4 ./ab.sh d_mtp_nopw
CFG=/tmp/basalt-prof/qwen-dflash2-nopw-baseline.json RUNS=4 ./ab.sh d_df2_nopw
CFG=/tmp/basalt-prof/qwen-dflash2-en-nopw-baseline.json RUNS=4 ./ab.sh d_df2en_nopw
} > batch15.out 2>&1
echo WINDOW3B DONE
