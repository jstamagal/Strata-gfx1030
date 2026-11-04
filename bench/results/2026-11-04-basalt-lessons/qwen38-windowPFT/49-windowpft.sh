#!/bin/bash
# Window PFT: where a fully resident 36K IQ3_S prompt spends its time (swarm ziz road 3). Live config, main build-hip.
# Control arm, then STRATA_PREFILL_TIMING=1 (per-phase GPU event timing; may perturb, so the control gives the real speed).
H=/home/ronald_rump/inbox/agents/claude/strata-ab
E=/home/ronald_rump/Strata_donttouch/build-hip/strata
cd "$H" || exit 1
echo "window PFT start $(date +%H:%M)"
(
    for arm in wpft_ctl wpft_timing; do
        if [ -e "$H/ab/$arm" ]; then echo "GATE FAIL: $H/ab/$arm exists"; exit 1; fi
    done
    sha256sum "$E"
    for arm in wpft_ctl wpft_timing; do
        extra=()
        [ "$arm" = wpft_timing ] && extra=("STRATA_PREFILL_TIMING=1")
        env CFG="$H/qwen-iq3s.json" MODEL=qwen3.8-abl-iq3_s PROMPT="$H/long-prompt.json" RUNS=2 MAXTOK=32 \
            "$H/pab2.sh" "$arm" "EXE=$E" "${extra[@]}"
        echo "HARNESS_STATUS $arm $?"
    done
    echo "--- prefill timing lines (wpft_timing engine.log)"
    grep -iE "prefill.*(timing|ms)|gemm|gdn|qsa|ple|combine|kv stage|hc " "$H/ab/wpft_timing/engine.log" | tail -60
) > "$H/windowpft.out" 2>&1
echo WINDOWPFT DONE
