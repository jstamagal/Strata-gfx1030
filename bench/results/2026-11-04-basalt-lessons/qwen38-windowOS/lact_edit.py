# usage: lact_edit.py pin|unpin  -- PF1 memory_clock power state [3] (pin top) or [1,2,3] (as found); always copies PF1's
# per-GPU settings into the top-level 'gpus' fallback (it had card0 at 150 W / no offset, card1 at 2600 MHz)
import sys, yaml, copy
p = '/etc/lact/config.yaml'
c = yaml.safe_load(open(p))
pf = c['profiles']['PF1']['gpus']
for g in pf.values():
    g.setdefault('power_states', {})['memory_clock'] = [3] if sys.argv[1] == 'pin' else [1, 2, 3]
c['gpus'] = copy.deepcopy(pf)
sys.stdout.write(yaml.safe_dump(c, sort_keys=False))
