#!/bin/bash
# IQ3_S cost-aware lookup16 sweep. Staged only; the coordinator launches it.
set -uo pipefail
H=/home/ronald_rump/inbox/agents/claude/strata-ab
E=/home/ronald_rump/inbox/agents/claude/strata-q38-splitlookup/build/strata
# 12179156f0b0c7f937dac30d772d1fe6cd612730: cost-aware admission plus stale-evidence recovery.
POLICY_COMMIT_TIME=1793821540
if [ ! -x "$E" ] || [ "$(stat -c %Y "$E" 2>/dev/null || printf 0)" -le "$POLICY_COMMIT_TIME" ]; then
    echo "WINDOWL16C SKIPPED: policy binary is absent or not newer than 12179156f"
    echo WINDOWL16C DONE
    exit 0
fi
CPU=/home/ronald_rump/inbox/agents/claude/strata-q38-splitlookup/build/cost-policy-cpu-check.log
if [ ! -f "$CPU" ] || [ "$(stat -c %Y "$CPU" 2>/dev/null || printf 0)" -lt "$(stat -c %Y "$E")" ] ||
   ! grep -Eq '^100% tests passed(, 0 tests failed)? out of 7[[:space:]]*$' "$CPU"; then
    echo "WINDOWL16C SKIPPED: this policy binary has not passed its seven CPU regressions"
    echo WINDOWL16C DONE
    exit 0
fi
EDIT=$H/edit-prompts.json
AGENT=/home/ronald_rump/inbox/agents/claude/strata-q38-dflash2/bench/results/2026-11-04-dflash2-agent-prompts.json
cd "$H" || exit 1
unset STRATA_PL_LOOKUP_MAX_T STRATA_LOOKUP_LONG_GATE
COMMON=("EXE=$E" "STRATA_PL_LOOKUP_MAX_T=" "STRATA_LOOKUP_LONG_GATE="
        "STRATA_LOOKUP_LONG_MIN_MATCH=32" "STRATA_LOOKUP_LONG_POLICY=1"
        "STRATA_LOOKUP_LONG_MIN_RUN=192" "STRATA_LOOKUP_LONG_HORIZON=512"
        "STRATA_LOOKUP_LONG_NARROW_MS=18.002" "STRATA_LOOKUP_LONG_WIDE_MS=33.175"
        "STRATA_LOOKUP_LONG_ENTRY_MS=13.414" "STRATA_LOOKUP_LONG_RECOVERY_MS=32.707"
        "STRATA_LOOKUP_LONG_PRIOR_FULL=0.99" "STRATA_LOOKUP_LONG_PRIOR_BONUS=0.99"
        "STRATA_LOOKUP_LONG_HISTORY=16" "STRATA_LOOKUP_LONG_MARGIN_MS=0"
        "STRATA_PIPELINE_DEBUG=1" "STRATA_PIPELINE_TRACE=pipeline.trace")
run() {
    local name=$1 prompts=$2
    shift 2
    env CFG="$H/qwen-iq3s.json" MODEL=qwen3.8-abl-iq3_s PROMPTS="$prompts" \
        "$H/agab2.sh" "$name" "${COMMON[@]}" "$@"
}
echo "window L16C start $(date +%H:%M)"
run_fail=0
{
    sha256sum "$E"
    run wl16c_edit_base     "$EDIT" || run_fail=1
    run wl16c_edit_p128     "$EDIT" "ARG=--lookup-max-t 16" "STRATA_LOOKUP_LONG_GATE=1" "STRATA_LOOKUP_LONG_MIN_RUN=128" || run_fail=1
    run wl16c_edit_p192     "$EDIT" "ARG=--lookup-max-t 16" "STRATA_LOOKUP_LONG_GATE=1" || run_fail=1
    run wl16c_edit_p256     "$EDIT" "ARG=--lookup-max-t 16" "STRATA_LOOKUP_LONG_GATE=1" "STRATA_LOOKUP_LONG_MIN_RUN=256" || run_fail=1
    run wl16c_edit_wide_off "$EDIT" "ARG=--lookup-max-t 16" "STRATA_LOOKUP_LONG_GATE=0" || run_fail=1
    run wl16c_agent_base     "$AGENT" || run_fail=1
    run wl16c_agent_p192     "$AGENT" "ARG=--lookup-max-t 16" "STRATA_LOOKUP_LONG_GATE=1" || run_fail=1
    # A fresh equal-geometry agent control is necessary for both requested text gates.
    run wl16c_agent_wide_off "$AGENT" "ARG=--lookup-max-t 16" "STRATA_LOOKUP_LONG_GATE=0" || run_fail=1
    python3 -I - "$H" "$EDIT" "$AGENT" "$run_fail" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1]) / 'ab'
failures = ['one or more A/B harness invocations failed'] if int(sys.argv[4]) else []
for group, prompts, policies in [('edit', sys.argv[2], {'p128': 128, 'p192': 192, 'p256': 256}),
                                  ('agent', sys.argv[3], {'p192': 192})]:
    cases = json.loads(Path(prompts).read_text())['cases']
    arms = ['base', *policies, 'wide_off']
    for case in cases:
        cid = case['id']
        texts, digests = {}, {}
        for arm in arms:
            name = f'wl16c_{group}_{arm}'
            try:
                doc = json.loads((root / name / f'out_{cid}.json').read_text())
                text = doc['choices'][0]['message']['content']
                if not isinstance(text, str):
                    raise ValueError('response content is not text')
                texts[arm] = text
                digests[arm] = hashlib.md5(text.encode()).hexdigest()
                print(f'MD5 {name} {cid} {digests[arm]}')
            except (OSError, KeyError, IndexError, TypeError, ValueError) as exc:
                failures.append(f'{name} {cid}: invalid/missing response: {exc}')
        comparisons = [(arm, ref) for arm in policies for ref in ('base', 'wide_off')]
        comparisons.append(('wide_off', 'base'))
        for arm, reference in comparisons:
            if arm not in texts or reference not in texts:
                print(f'MD5_COMPARE {group} {cid} {arm} vs {reference}: UNAVAILABLE')
                continue
            same = texts[arm] == texts[reference]
            print(f'MD5_COMPARE {group} {cid} {arm} vs {reference}: {"MATCH" if same else "DIFF"} '
                  f'{digests[arm]} {digests[reference]}')
            if not same:
                failures.append(f'{group} {cid}: {arm} differs from {reference}')
    for arm in arms:
        name = f'wl16c_{group}_{arm}'
        try:
            log = (root / name / 'engine.log').read_text(errors='replace')
            windows = sum(map(int, re.findall(r'strata serve: long lookup: (\d+) windows', log)))
            print(f'LONG_LOOKUP {name} {windows} windows')
            if arm not in policies and windows:
                failures.append(f'{name}: control admitted long lookup windows')
            if arm in policies:
                records = re.findall(r'lookup cost: policy (on|off), min run (\d+), horizon (\d+); '
                                     r'(\d+) checks, (\d+) admitted, (\d+) declined', log)
                if not records:
                    failures.append(f'{name}: no cost-policy instrumentation (wrong binary/config)')
                elif any(on != 'on' or int(threshold) != policies[arm] or int(horizon) != 512
                         for on, threshold, horizon, *_ in records):
                    failures.append(f'{name}: cost-policy settings do not match this sweep arm')
                checks = sum(int(r[3]) for r in records)
                admitted = sum(int(r[4]) for r in records)
                declined = sum(int(r[5]) for r in records)
                print(f'LOOKUP_COST {name} min_run={policies[arm]} checks={checks} '
                      f'admitted={admitted} declined={declined}')
                if group == 'edit' and checks == 0:
                    failures.append(f'{name}: cost model was never exercised')
                if windows == 0:
                    print(f'POLICY_OUTCOME {name}: zero wide windows; valid narrow fallback, not evidence of wide speedup')
        except OSError as exc:
            failures.append(f'{name}: missing engine log: {exc}')
if failures:
    for failure in failures:
        print('GATE FAIL:', failure)
    print('WINDOWL16C VALIDATION FAILED: do not adopt')
else:
    print('WINDOWL16C VALIDATION PASSED: full text/MD5 matches base and equal-geometry control; edit policy exercised')
sys.exit(bool(failures))
PY
} > "$H/windowl16c.out" 2>&1
status=$?
echo WINDOWL16C DONE
exit "$status"
