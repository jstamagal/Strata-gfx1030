#!/bin/bash
# IQ3_S integration: today's engine vs next, route-ahead prompt, fresh-chain cap.
# Staged only: the coordinator moves this into the queue after the build slot.
H=/home/ronald_rump/inbox/agents/claude/strata-ab
CUR=/home/ronald_rump/Strata_donttouch/build-hip/strata
NEXT=/home/ronald_rump/inbox/agents/claude/strata-q38-next/build/strata
CFG=$H/qwen-iq3s.json
MODEL=qwen3.8-abl-iq3_s
AGENT=/home/ronald_rump/inbox/agents/claude/strata-q38-dflash2/bench/results/2026-11-04-dflash2-agent-prompts.json
EDIT=$H/edit-prompts.json
LONG=$H/long-prompt.json
cd "$H" || exit 1
# Clear inherited experiment settings as well as removing their config overrides.
unset STRATA_MTP_CHAIN_CAP STRATA_PF_ROUTE_AHEAD STRATA_PL_FRESH_T STRATA_PL_GUESS_T
COMMON=("ARG=--mtp-max-t 3" "ARG=--prefill 4096" "STRATA_MTP_CHAIN_CAP=0"
        "STRATA_PF_ROUTE_AHEAD=" "STRATA_PL_FRESH_T=" "STRATA_PL_GUESS_T=")
harness_failures=0
run_agent() {
    local name=$1 exe=$2 prompts=$3 rc
    shift 3
    env CFG="$CFG" MODEL="$MODEL" PROMPTS="$prompts" \
        "$H/agab2.sh" "$name" "EXE=$exe" "${COMMON[@]}" "$@"
    rc=$?
    echo "HARNESS_STATUS $name $rc"
    if [ "$rc" -ne 0 ]; then harness_failures=$((harness_failures + 1)); fi
}
run_prompt() {
    local name=$1 exe=$2 rc
    shift 2
    env CFG="$CFG" MODEL="$MODEL" PROMPT="$LONG" RUNS=2 MAXTOK=32 \
        "$H/pab2.sh" "$name" "EXE=$exe" "${COMMON[@]}" "$@"
    rc=$?
    echo "HARNESS_STATUS $name $rc"
    # pab2's last command waits for its intentionally terminated server (143).
    # The response/request checks below still reject missing or invalid runs.
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 143 ]; then
        harness_failures=$((harness_failures + 1))
    fi
}
echo "window NEXT start $(date +%H:%M)"
(
    echo "CUR=$CUR NEXT=$NEXT CFG=$CFG MODEL=$MODEL"
    if [ ! -x "$CUR" ] || [ ! -x "$NEXT" ]; then
        echo 'GATE FAIL: control or next binary is missing/not executable'
        exit 1
    fi
    for arm in wn_cur_agent wn_next_agent wn_chain_agent wn_cur_prompt wn_next_prompt \
               wn_ahead_prompt wn_next_edit wn_chain_edit; do
        if [ -e "$H/ab/$arm" ]; then
            echo "GATE FAIL: $H/ab/$arm already exists; refusing stale responses"
            exit 1
        fi
    done
    sha256sum "$CUR" "$NEXT"
    git -C /home/ronald_rump/inbox/agents/claude/strata-q38-next rev-parse HEAD
    run_agent  wn_cur_agent    "$CUR"  "$AGENT"
    run_agent  wn_next_agent   "$NEXT" "$AGENT"
    run_agent  wn_chain_agent  "$NEXT" "$AGENT" "STRATA_MTP_CHAIN_CAP=1"
    run_prompt wn_cur_prompt   "$CUR"
    run_prompt wn_next_prompt  "$NEXT"
    run_prompt wn_ahead_prompt "$NEXT" "STRATA_PF_ROUTE_AHEAD=0,32"
    # A next edit control is needed to isolate the chain-cap edit arm.
    run_agent  wn_next_edit    "$NEXT" "$EDIT"
    run_agent  wn_chain_edit   "$NEXT" "$EDIT" "STRATA_MTP_CHAIN_CAP=1"
    # Enforce full content and identical requests, not the harness's short digest.
    python3 -I - "$H" "$AGENT" "$EDIT" "$harness_failures" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]) / 'ab'
failures = []
if int(sys.argv[4]):
    failures.append(f'{sys.argv[4]} harness invocation(s) failed')


def load(arm, filename):
    return json.loads((root / arm / filename).read_text())


def response(arm, filename, case):
    doc = load(arm, filename)
    text = doc['choices'][0]['message']['content']
    if not isinstance(text, str):
        raise ValueError('response content is not text')
    timing = doc['timings']
    if timing['predicted_n'] <= 0 or timing['predicted_per_second'] <= 0:
        raise ValueError('missing/nonpositive generation timing')
    digest = hashlib.md5(text.encode('utf-8')).hexdigest()
    print(f'MD5 {arm} {case} {digest}')
    return text


def compare(group, cases, arms, output_file, request_file):
    for case in cases:
        reference = request = None
        for arm in arms:
            try:
                text = response(arm, output_file(case), case)
                req = load(arm, request_file(case))
                if arm == arms[0]:
                    reference, request = text, req
                else:
                    if reference is None or text != reference:
                        failures.append(f'{group} {arm} {case}: full text differs from {arms[0]}')
                    if request is None or req != request:
                        failures.append(f'{group} {arm} {case}: request differs from {arms[0]}')
            except (OSError, KeyError, IndexError, TypeError, ValueError) as exc:
                failures.append(f'{group} {arm} {case}: invalid/missing response or request: {exc}')


for group, path, arms in [
    ('agent', sys.argv[2], ['wn_cur_agent', 'wn_next_agent', 'wn_chain_agent']),
    ('edit', sys.argv[3], ['wn_next_edit', 'wn_chain_edit']),
]:
    try:
        cases = [case['id'] for case in json.loads(Path(path).read_text())['cases']]
        if not cases or len(cases) != len(set(cases)):
            raise ValueError('empty or duplicate case IDs')
        compare(group, cases, arms, lambda c: f'out_{c}.json', lambda c: f'req_{c}.json')
    except (OSError, KeyError, TypeError, ValueError) as exc:
        failures.append(f'{group}: invalid prompts: {exc}')
compare('36K prompt', [1, 2], ['wn_cur_prompt', 'wn_next_prompt', 'wn_ahead_prompt'],
        lambda c: f'long{c}.json', lambda c: f'req{c}.json')
if failures:
    for failure in failures:
        print('GATE FAIL:', failure)
    print('NUMERICS AUDIT: basalt38/rdna2-mmq (0cae8429c) changes prompt/decode kernel '
          'lowering and tile dispatch, including IQ4_NL PERM at TJ=20 used by this config. '
          'Its optional STRATA_MMQ_Q8K / STRATA_PF_IQ_Q8K paths change activation quantization; '
          'they are not enabled by this script.')
    print('NUMERICS AUDIT: head Q6 proxy pick 1e7376504 can change draft IDs/probabilities '
          'when auto becomes active; it must not change target logits. This CFG already '
          'forces STRATA_MTP_HEAD_PROXY=1, so the auto-policy change should not affect these arms.')
    print('NUMERICS AUDIT: basalt38/tp-prompt route-ahead is a copy hint, and '
          'STRATA_MTP_CHAIN_CAP only omits unused outputs. Neither is intended to change '
          'emitted text. A mismatch is a failed gate, not a proven cause or permission to adopt.')
    print('WINDOWNEXT VALIDATION FAILED: do not adopt')
else:
    print('WINDOWNEXT VALIDATION PASSED: all full-text MD5s and requests match their controls')
sys.exit(bool(failures))
PY
) > "$H/batch35.out" 2>&1
status=$?
echo WINDOWNEXT DONE
exit "$status"
