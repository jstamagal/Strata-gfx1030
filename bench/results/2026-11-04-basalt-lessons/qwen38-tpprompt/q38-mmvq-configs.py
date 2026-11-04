import json
from pathlib import Path
root = Path('/home/ronald_rump/inbox/agents/claude/strata-q38-tpprompt')
out = root / 'bench/results/2026-11-04-basalt-lessons/qwen38-tpprompt'
base = json.loads((out / 'pf2048-control.json').read_text())
for ceiling in [0, 32, 64, 128, 256]:
    c = json.loads(json.dumps(base))
    name = f'mmvq-max{ceiling}' if ceiling else 'mmvq-control'
    c['exe'] = str(root / 'build/strata-pf-mmvq-4158')
    c['log'] = str(out / 'out' / (name + '.log'))
    c['env'].update(STRATA_PF_MMVQ='1' if ceiling else '0', STRATA_PF_MMVQ_RESERVE='1',
                    STRATA_PF_MMVQ_MAX_T=str(ceiling or 256), STRATA_PF_MMVQ_LOG='1')
    (out / (name + '.json')).write_text(json.dumps(c, indent=2) + '\n')
    print(out / (name + '.json'))
