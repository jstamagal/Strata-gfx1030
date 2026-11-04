#!/bin/bash
# IQ3_S rerun after the standalone ROW/SET dispatcher fix; no wait and no automatic launch.
H=/home/ronald_rump/inbox/agents/claude/strata-ab
E=/home/ronald_rump/inbox/agents/claude/strata-q38-splitlookup/build/strata
# 27d713fbcb4a8808f68a349887b723d26864b918, committed 2026-11-04T13:14:07-05:00.
FIX_COMMIT_TIME=1793816047
if [ ! -x "$E" ] || [ "$(stat -c %Y "$E" 2>/dev/null || printf 0)" -le "$FIX_COMMIT_TIME" ]; then
    echo "WINDOWL16B SKIPPED: fixed binary is absent or not newer than dispatcher fix 27d713fbc"
    echo WINDOWL16B DONE
    exit 0
fi
EDIT=$H/edit-prompts.json
AGENT=/home/ronald_rump/inbox/agents/claude/strata-q38-dflash2/bench/results/2026-11-04-dflash2-agent-prompts.json
cd "$H" || exit 1
unset STRATA_PL_LOOKUP_MAX_T STRATA_LOOKUP_LONG_GATE
COMMON=("EXE=$E" "STRATA_PL_LOOKUP_MAX_T=" "STRATA_LOOKUP_LONG_GATE="
        "STRATA_LOOKUP_LONG_MIN_MATCH=32" "STRATA_PIPELINE_DEBUG=1" "STRATA_PIPELINE_TRACE=pipeline.trace")
run() {
    local name=$1 prompts=$2
    shift 2
    env CFG="$H/qwen-iq3s.json" MODEL=qwen3.8-abl-iq3_s PROMPTS="$prompts" \
        "$H/agab2.sh" "$name" "${COMMON[@]}" "$@"
}
echo "window L16B start $(date +%H:%M)"
{
    sha256sum "$E"
    run wl16b_edit_base      "$EDIT"
    run wl16b_edit_long      "$EDIT" "ARG=--lookup-max-t 16" "STRATA_LOOKUP_LONG_GATE=1"
    run wl16b_edit_wide_off  "$EDIT" "ARG=--lookup-max-t 16" "STRATA_LOOKUP_LONG_GATE=0"
    run wl16b_agent_base     "$AGENT"
    run wl16b_agent_long     "$AGENT" "ARG=--lookup-max-t 16" "STRATA_LOOKUP_LONG_GATE=1"
    run wl16b_agent_wide_off "$AGENT" "ARG=--lookup-max-t 16" "STRATA_LOOKUP_LONG_GATE=0"
    python3 -I - "$H" "$EDIT" "$AGENT" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1]) / 'ab'
failures = []
for group, prompts in [('edit', sys.argv[2]), ('agent', sys.argv[3])]:
    cases = json.loads(Path(prompts).read_text())['cases']
    arms = ['base', 'long', 'wide_off']
    for case in cases:
        cid = case['id']
        texts = {}
        digests = {}
        for arm in arms:
            name = f'wl16b_{group}_{arm}'
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
        # Compare the wide arms to base, and long to its equal-allocation control.
        for arm, reference in [('long', 'base'), ('long', 'wide_off'), ('wide_off', 'base')]:
            if arm not in texts or reference not in texts:
                print(f'MD5_COMPARE {group} {cid} {arm} vs {reference}: UNAVAILABLE')
                continue
            same = texts[arm] == texts[reference]
            result = 'MATCH' if same else 'DIFF'
            print(f'MD5_COMPARE {group} {cid} {arm} vs {reference}: {result} '
                  f'{digests[arm]} {digests[reference]}')
            if not same:
                failures.append(f'{group} {cid}: {arm} differs from {reference}')
    for arm in arms:
        name = f'wl16b_{group}_{arm}'
        try:
            log = (root / name / 'engine.log').read_text(errors='replace')
            windows = sum(map(int, re.findall(r'strata serve: long lookup: (\d+) windows', log)))
            print(f'LONG_LOOKUP {name} {windows} windows')
            if arm != 'long' and windows:
                failures.append(f'{name}: control admitted long lookup windows')
            if group == 'edit' and arm == 'long' and not windows:
                failures.append(f'{name}: no long lookup window activated')
        except OSError as exc:
            failures.append(f'{name}: missing engine log: {exc}')
if failures:
    for failure in failures:
        print('GATE FAIL:', failure)
    print('WINDOWL16B VALIDATION FAILED: do not adopt')
else:
    print('WINDOWL16B VALIDATION PASSED: full text/MD5 matches base and equal-geometry control; edit long activated')
sys.exit(bool(failures))
PY
} > "$H/batch37.out" 2>&1
echo WINDOWL16B DONE
