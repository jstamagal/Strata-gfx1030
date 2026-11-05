#!/bin/bash
# Queue window BIM; DO NOT run outside the coordinator's GPU queue. Uses BOTH V620s.
# Runtime estimate: 22-25 minutes (12 agent loads ~68 s + 4 edit24 loads ~128 s, plus gates/readout).
# Agent passes: m u f c / c f u m / m u f c. Edit24: one m u f c pass. Every load is a fresh server.
# All measurement outputs are ab/bim_*. Existing bim_* entries refuse the whole window; nothing is overwritten.
set -euo pipefail
H=/home/ronald_rump/inbox/agents/claude/strata-ab
AB=$H/ab
R=/home/ronald_rump/Strata_donttouch
W=/home/ronald_rump/inbox/agents/claude/strata-q38-bimodal
B=$W/build-hip
MAIN=$R/build-hip/strata
BRANCH=$B/strata
READER=$W/tools/hip/bimodal-read.py
BASECFG=$AB/prport2-live-cfg.json
AGENT=/home/ronald_rump/inbox/agents/claude/strata-q38-dflash2/bench/results/2026-11-04-dflash2-agent-prompts.json
EDIT24=/home/ronald_rump/inbox/agents/claude/mtpq/prompts/edit24.json
FAST=$AB/prp2_m_agent_2
SLOW=$AB/prp2_m_agent_1
EDITREF=$AB/prp2_m_edit_2
PROFILE=14.3505,15.5044,17.9436,24.1993,28.9211,33.6429,38.3647,24.9153
CLIP=2.10
SELF=$(readlink -f "${BASH_SOURCE[0]}")
fail() { echo "GATE FAIL: $*"; exit 1; }
shopt -s nullglob
existing=("$AB"/bim_*)
[ "${#existing[@]}" -eq 0 ] || fail 'ab/bim_* already exists; refusing to start'
for f in "$MAIN" "$BRANCH" "$READER" "$BASECFG" "$AGENT" "$EDIT24" "$H/agab2.sh" \
         "$B/bimodal-build.commit" "$B/bimodal-build.sha256" "$FAST/pipeline.trace" "$SLOW/pipeline.trace" "$EDITREF/pipeline.trace"; do
    [ -f "$f" ] || fail "missing input: $f"
done
HEAD=$(git -C "$W" rev-parse HEAD)
[ "$(git -C "$W" branch --show-current)" = basalt38/bimodal ] || fail 'wrong worktree branch'
[ "$(cat "$B/bimodal-build.commit")" = "$HEAD" ] || fail 'build commit differs from branch HEAD'
[ -z "$(git -C "$W" status --porcelain --untracked-files=no)" ] || fail 'tracked worktree is dirty'
sha256sum -c --quiet "$B/bimodal-build.sha256" || fail 'built binary changed after its receipt'
# Atomic ownership marker also closes the race between two simultaneous invocations.
mkdir "$AB/bim_lock" || fail 'another BIM invocation claimed the output prefix'
set +e
(
    set -e
    trap 'rc=$?; if [ -f "$AB/bim_inputs.sha256" ]; then sha256sum -c --quiet "$AB/bim_inputs.sha256" || rc=1; fi; if [ "$rc" -ne 0 ]; then echo "WINDOWBIMODAL CHECK FAILED (window exit $rc)"; fi; exit "$rc"' EXIT
    echo "window BIM start $(date -Is) branch=$HEAD cards=0,1 profile=$PROFILE clip=$CLIP"
    CFG=$AB/bim_live_cfg.json
    cp "$BASECFG" "$CFG"
    export TMPDIR=$W/scratch/bimodal
    mkdir -p "$TMPDIR"
    python3 -I - "$CFG" "$AGENT" "$EDIT24" "$FAST" "$EDITREF" <<'PY'
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
assert cfg['gpu'] == [0, 1] and str(cfg['layer_split']) == '25', 'both-card split changed'
a = cfg['args']
assert a[a.index('--pipeline-windows') + 1] == '2', 'pipeline width changed'
assert a[a.index('--mtp') + 1] == '/home/ronald_rump/Strata-data/mtp/q38-s2-gptq-a', 'live drafter changed'
for prompt_path, ref, expected in ((sys.argv[2], sys.argv[4], 6), (sys.argv[3], sys.argv[5], 24)):
    cases = json.loads(Path(prompt_path).read_text())['cases']
    assert len(cases) == expected and len({x['id'] for x in cases}) == expected
    for c in cases:
        req = json.loads((Path(ref) / ('req_' + c['id'] + '.json')).read_text())
        assert req['messages'] == c['messages'], 'prompts differ from saved reference'
        assert req['temperature'] == 0 and req['max_tokens'] == 1200
        assert req['chat_template_kwargs']['enable_thinking'] is False
PY
    python3 -I - "$AB/bim_manifest.json" "$MAIN" "$BRANCH" "$PROFILE" "$CLIP" <<'PY'
import json, sys
loads = []
for p, arms in ((1, 'mufc'), (2, 'cfum'), (3, 'mufc')):
    loads += [dict(name=f'bim_{a}_agent_p{p}', arm=a, kind='agent', **{'pass': p}) for a in arms]
loads += [dict(name=f'bim_{a}_edit24_p1', arm=a, kind='edit24', **{'pass': 1}) for a in 'mufc']
with open(sys.argv[1], 'w') as f:
    json.dump(dict(main_exe=sys.argv[2], branch_exe=sys.argv[3], profile=sys.argv[4], clip=sys.argv[5], loads=loads), f, indent=2)
PY
    inputs=("$MAIN" "$BRANCH" "$CFG" "$BASECFG" "$AGENT" "$EDIT24" "$H/agab2.sh" "$READER" "$SELF"
            "$B/bimodal-build.commit" "$B/bimodal-build.sha256" "$AB/bim_manifest.json"
            "$FAST/pipeline.trace" "$SLOW/pipeline.trace" "$EDITREF/pipeline.trace"
            "$FAST"/out_*.json "$SLOW"/out_*.json "$EDITREF"/out_*.json)
    sha256sum "${inputs[@]}" > "$AB/bim_inputs.sha256"
    : > "$AB/bim_harness.tsv"
    arm() {
        local v=$1 kind=$2 name=bim_$1_$2_p$3 prompts=$AGENT exe=$BRANCH fixed='' clip='' rc
        [ "$kind" = edit24 ] && prompts=$EDIT24
        [ "$v" = m ] && exe=$MAIN
        [ "$v" = f ] && fixed=$PROFILE
        [ "$v" = c ] && clip=$CLIP
        [ ! -e "$AB/$name" ] || fail "load output already exists: $name"
        echo "arm $name start $(date -Is) EXE=$exe FIXED=$fixed CLIP=$clip"
        set +e
        env -u STRATA_PL_POLICY_COSTS -u STRATA_PL_POLICY_CLIP CFG="$CFG" MODEL=qwen3.8-abl-iq3_s PROMPTS="$prompts" \
            "$H/agab2.sh" "$name" "EXE=$exe" 'CFG.host=127.0.0.1' \
            'STRATA_PIPELINE_DEBUG=1' 'STRATA_PIPELINE_TRACE=pipeline.trace' \
            "STRATA_PL_POLICY_COSTS=$fixed" "STRATA_PL_POLICY_CLIP=$clip" > "$AB/${name}.pending" 2>&1
        rc=$?
        set -e
        mkdir -p "$AB/$name"
        mv "$AB/${name}.pending" "$AB/$name/harness.out"
        printf '%s\t%d\n' "$name" "$rc" >> "$AB/bim_harness.tsv"
        echo "HARNESS_STATUS $name $rc"
        [ "$rc" -eq 0 ] || fail "$name: harness failed; no further loads started"
    }
    for v in m u f c; do arm "$v" agent 1; done
    for v in c f u m; do arm "$v" agent 2; done
    for v in m u f c; do arm "$v" agent 3; done
    for v in m u f c; do arm "$v" edit24 1; done
    [ "$(git -C "$W" rev-parse HEAD)" = "$HEAD" ] || fail 'branch advanced during the window'
    [ -z "$(git -C "$W" status --porcelain --untracked-files=no)" ] || fail 'tracked worktree changed during the window'
    sha256sum -c --quiet "$AB/bim_inputs.sha256" || fail 'an input changed during the window'
    python3 -I "$READER" read "$AB" "$AB/bim_manifest.json" "$FAST" "$SLOW" "$EDITREF" | tee "$AB/bim_readout.txt"
    echo "window BIM end $(date -Is)"
) > "$AB/bim_window.out" 2>&1
status=$?
set -e
echo "WINDOWBIMODAL DONE exit $status; output $AB/bim_window.out"
exit "$status"
