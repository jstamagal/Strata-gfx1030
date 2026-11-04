import json
from pathlib import Path
root = Path('/home/ronald_rump/inbox/agents/claude/strata-q38-tpprompt')
out = root / 'bench/results/2026-11-04-basalt-lessons/qwen38-tpprompt'
out.mkdir(parents=True, exist_ok=True)
(out / 'out').mkdir(exist_ok=True)
source = Path('/home/ronald_rump/Strata_donttouch/strata-unsloth-qwen-UD4_K_XL.json')
base = json.loads(source.read_text())

def make(name, prefill, env, split='26'):
    c = json.loads(json.dumps(base))
    c['exe'] = str(root / 'build/strata-tpprompt-06a0e3fec')
    c['layer_split'] = split
    c['port'] = 8091
    c['host'] = '127.0.0.1'
    c['log'] = str(out / 'out' / (name + '.log'))
    args = c['args']
    args[args.index('--prefill') + 1] = str(prefill)
    # Freeze placement for the numerical A/B. This changes no first cold prompt arithmetic.
    args[args.index('--adapt-every') + 1] = '0'
    c['env'].update(STRATA_PF_ROUTE_AHEAD='0', STRATA_LM_SPLIT_R_VRAM='0', STRATA_PREFILL_LAYER_MAJOR='0')
    c['env'].update(env)
    (out / (name + '.json')).write_text(json.dumps(c, indent=2) + '\n')

for chunk in [2048, 4096]:
    make(f'pf{chunk}-control', chunk, {})
    for n in [16, 32]:
        make(f'pf{chunk}-ahead{n}', chunk, {'STRATA_PF_ROUTE_AHEAD': f'0,{n}'})
for superchunk in [4096, 8192]:
    lm = {'STRATA_PREFILL_LAYER_MAJOR': '1', 'STRATA_LM_HIP': '1',
          'STRATA_PREFILL_PIPE': '0', 'STRATA_PREFILL_LM_TOKENS': str(superchunk),
          'STRATA_PREFILL_LM_R': 'vram'}
    make(f'lm{superchunk}-host-later', 2048, lm)
    make(f'lm{superchunk}-vram-later', 2048, dict(lm, STRATA_LM_SPLIT_R_VRAM='1'))
print(out)
for p in sorted(out.glob('*.json')):
    json.loads(p.read_text())
    print(p.name)
