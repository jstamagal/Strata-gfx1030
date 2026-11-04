#!/bin/bash
# IQ3_S split lookup16: three edit arms and two agent arms, all using the same new binary.
# No wait: the coordinator starts this only after the build and CPU checks are complete.
H=/home/ronald_rump/inbox/agents/claude/strata-ab
E=/home/ronald_rump/inbox/agents/claude/strata-q38-splitlookup/build/strata
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
echo "window L16 start $(date +%H:%M)"
{
    sha256sum "$E"
    run wl16_edit_base     "$EDIT"
    run wl16_edit_long     "$EDIT" "ARG=--lookup-max-t 16" "STRATA_LOOKUP_LONG_GATE=1"
    run wl16_edit_wide_off "$EDIT" "ARG=--lookup-max-t 16" "STRATA_LOOKUP_LONG_GATE=0"
    run wl16_agent_base    "$AGENT"
    run wl16_agent_long    "$AGENT" "ARG=--lookup-max-t 16" "STRATA_LOOKUP_LONG_GATE=1"
    # agab2 displays a short digest; enforce the full generated text / MD5 gate here.
    python3 -I - "$H" "$EDIT" "$AGENT" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1]) / 'ab'
failures = []
for group, prompts, arms in [('edit', sys.argv[2], ['base', 'long', 'wide_off']),
                             ('agent', sys.argv[3], ['base', 'long'])]:
    cases = json.loads(Path(prompts).read_text())['cases']
    for case in cases:
        cid = case['id']
        reference = None
        for arm in arms:
            name = f'wl16_{group}_{arm}'
            try:
                doc = json.loads((root / name / f'out_{cid}.json').read_text())
                text = doc['choices'][0]['message']['content']
                if not isinstance(text, str):
                    raise ValueError('response content is not text')
                digest = hashlib.md5(text.encode()).hexdigest()
                print(f'MD5 {name} {cid} {digest}')
                if arm == 'base':
                    reference = text
                elif reference is None or text != reference:
                    failures.append(f'{name} {cid}: output differs from base')
            except (OSError, KeyError, IndexError, TypeError, ValueError) as exc:
                failures.append(f'{name} {cid}: invalid/missing response: {exc}')
    for arm in arms:
        name = f'wl16_{group}_{arm}'
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
    print('WINDOWL16 VALIDATION FAILED: do not adopt')
else:
    print('WINDOWL16 VALIDATION PASSED: full text/MD5 matches; edit long windows activated')
sys.exit(bool(failures))
PY
} > "$H/batch34.out" 2>&1
echo WINDOWL16 DONE
