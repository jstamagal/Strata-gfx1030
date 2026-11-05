#!/usr/bin/env python3
"""CPU-only synthetic checks; never configures HIP, starts the builder, launches an engine or queries GPUs."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import time
import unittest

spec = importlib.util.spec_from_file_location('window', Path(__file__).with_name('live_window.py'))
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
b = w.build


def log(arm):
    txt = ('CUDA1 runs layers 25-47, expert cache 11776 slots\n100% of the experts resident\n' +
           'zero-doorbell graph\n' * 4 + 'strata serve: 2182 MiB of VRAM free with everything loaded\n'
           'strata serve: CUDA1: 1388 MiB of VRAM free with everything loaded\n')
    if arm == 'L':
        txt += 'STRATA_T1_DIET 0xc: HC norm in the down launch off, one-row QSA batched on, DPP routing on\n' * 4
        txt += 'strata serve: STRATA_PL_POLICY_CLIP=2.1: upper cost samples clipped before EMA\n'
        for c, start, end, n, mib in ((0, 0, 24, 100, 351), (1, 25, 47, 92, 323)):
            txt += (f'STRATA_HC_REQ8 on a layer split: CUDA{c} layers {start}-{end}: {n} hyper-connection projections '
                    f'get an int8 + fp32/32 copy for the verify window ({mib} MiB; the BF16 rows stay for the prompt path) '
                    'in 0.6 s; 28.00 GiB free\n')
    return txt


def trace(ids, arm):
    lines = ['# request p0=0', '0 L0 0 1 0', '1 F0 0 0 0', '2 L1 0 1 0', '3 F1 0 0 0', '4 V 0 0 1']
    for cid in sorted(ids):
        lines.append('# request p0=100')
        x = 0.
        for seq, t in enumerate((1, 2, 3, 1, 2, 3)):
            d0 = 7 + t - (0.5 if arm == 'L' else 0)
            d1 = 8 + t - (0.5 if arm == 'L' else 0)
            lines += [f'{x} L0 {seq} {t} 0', f'{x+d0} F0 {seq} 0 0', f'{x+d0+.1} L1 {seq} {t} 0',
                      f'{x+d0+.1+d1} F1 {seq} 0 0', f'{x+d0+.2+d1} V {seq} {t-1} 1']
            x += d0 + d1 + 1
    return '\n'.join(lines) + '\n'


def response(ms=120, text='same'):
    return {'timings': {'predicted_n': 12, 'predicted_ms': ms, 'draft_n_accepted': 6, 'draft_n': 8},
            'choices': [{'message': {'content': text}}]}


class Live(unittest.TestCase):
    def test_plan_is_fully_mirrored_and_nine_fresh_loads(self):
        p = w.plan()
        self.assertEqual(len(p), 21)
        first = [(r['arm'], r['kind']) for r in p[:6]]
        second = [(r['arm'], r['kind']) for r in p[6:12]]
        self.assertEqual(second, list(reversed(first)))
        for arm in w.ARMS:
            self.assertEqual(sum(r['arm'] == arm and r['kind'] == 'read36' for r in p), 3)

    def test_default_env_exact_and_L_exact_delta(self):
        live = json.loads((b.R / 'strata-sc117-IQ3_S-qwen-abliterated.json').read_text())
        original = copy.deepcopy(live)
        for arm in w.ARMS:
            c = w.config(live, arm, Path('/fixture'))
            env = dict(c['env'])
            env.pop('STRATA_PIPELINE_DEBUG'); env.pop('STRATA_PIPELINE_TRACE')
            self.assertEqual(env, live['env'] | (b.L_ENV if arm == 'L' else {}))
            self.assertEqual(c['args'], live['args'])
        self.assertEqual(live, original)

    def test_gates_are_strict(self):
        for arm in w.ARMS:
            self.assertFalse(w.gate(log(arm), arm)['failures'])
        self.assertTrue(w.gate(log('L').replace('11776 slots', '11730 slots'), 'L')['failures'])
        self.assertTrue(w.gate(log('L').replace('fp32/32', 'fp16/16'), 'L')['failures'])
        self.assertTrue(w.gate(log('L').replace('2.1:', '2.2:'), 'L')['failures'])
        self.assertTrue(w.gate(log('L'), 'L0')['failures'])

    def test_agent_gate_prefers_shared_state_to_confounded_pool(self):
        metrics = {'agent/L': {'median_pct': 7.0}}
        state = {'L': {'matched': {'fast': {'median_pct': 1.0}, 'slow': {'median_pct': 2.0}}}}
        value, basis = w.agent_gate_metric(metrics, state)
        self.assertEqual(value, 1.0)
        self.assertIn('within-fast', basis)
        self.assertFalse(w.verdict([], True, {f'{k}/{s}': (7., 8.) for k in ('agent', 'edit24') for s in w.STAGES}, value, 1., 0.)[0])
        value, basis = w.agent_gate_metric(metrics, {})
        self.assertEqual(value, 7.0)
        self.assertIn('STATE-CONFOUNDED', basis)

    def test_verdict_boundaries_and_each_hold_reason(self):
        good = {f'{kind}/{key}': (7., 8.) for kind in ('agent', 'edit24') for key in w.STAGES}
        self.assertTrue(w.verdict([], True, good, 3., 0., -1.)[0])
        for args in ((['bad residency'], True, good, 4., 1., 0.), ([], False, good, 4., 1., 0.),
                     ([], True, good, 2.99, 1., 0.), ([], True, good, 4., -.01, 0.),
                     ([], True, good, 4., 1., -1.01), ([], True, {}, 4., 1., 0.)):
            go, reasons = w.verdict(*args)
            self.assertFalse(go); self.assertTrue(reasons)
        slower = dict(good); slower['agent/s1_T3'] = (8.1, 8.)
        self.assertFalse(w.verdict([], True, slower, 4., 1., 0.)[0])
        missing = dict(good); missing['agent/s1_T3'] = (None, 8.)
        self.assertFalse(w.verdict([], True, missing, 4., 1., 0.)[0])

    def test_output_prefix_refusal(self):
        root = b.S / ('prefix-fixture-' + str(time.time_ns()))
        root.mkdir(parents=True)
        w.reserve(root)
        with self.assertRaises(RuntimeError):
            w.reserve(root)

    def test_slot_guard_without_starting_builder(self):
        old = b.H
        root = b.S / ('slot-fixture-' + str(time.time_ns()))
        root.mkdir(parents=True)
        b.H = root
        try:
            with self.assertRaises(RuntimeError):
                b.slot()
            (root / 'buildslot.open').touch()
            (root / 'buildslot.expect').write_text('owner\n')
            with self.assertRaises(RuntimeError):
                b.slot()
            (root / 'buildslot.expect').write_text('owner live\n')
            b.slot()  # only the guard, NOT build()
        finally:
            b.H = old

    def test_state_classification_does_not_use_absolute_speed(self):
        def load(arm, hit, miss, tps, text='same'):
            return dict(arm=arm, hit=hit, miss=miss, tps=tps, cases={'c': {'text': text}})
        main = [load('m', 1898, 997, 151.6), load('m', 1750, 1030, 148.4)]
        off = [load('L0', 1898, 997, 155), load('L0', 1750, 1030, 152)]
        changed = [load('L', 2200, 1000, 170, 'changed'), load('L', 2000, 1030, 166, 'changed')]
        w.classify(main + off + changed, main)
        self.assertEqual([r['state'] for r in off], ['fast', 'slow'])
        self.assertEqual([r['state'] for r in changed], ['fast', 'slow'])
        changed[1].update(hit=2200, miss=1000, tps=170.1)
        w.classify(main + off + changed, main)
        self.assertEqual([r['state'] for r in changed], ['unknown', 'unknown'])

    def test_full_reader_go_and_corrupted_loads_hold(self):
        root = b.S / ('reader-fixture-' + str(time.time_ns()))
        root.mkdir(parents=True)
        (root / 'SYNTHETIC.txt').write_text('CPU fixtures only, NO GPU measurements.\n')
        live = json.loads((b.R / 'strata-sc117-IQ3_S-qwen-abliterated.json').read_text())
        ids = {'agent': [f'a{i}' for i in range(6)], 'edit24': [f'e{i:02}' for i in range(24)]}
        (root / 'live_manifest.json').write_text(json.dumps({'loads': w.plan(), 'ids': ids, 'hashes': {}}))
        (root / 'live_cfg.json').write_text(json.dumps(live))
        for item in w.plan():
            d = root / item['name']; d.mkdir()
            arm, kind = item['arm'], item['kind']
            (d / 'cfg.json').write_text(json.dumps(w.config(live, arm, d)))
            (d / 'status').write_text('0\n')
            (d / 'engine.log').write_text(log(arm))
            if kind == 'read36':
                j = {'timings': {'prompt_n': 36000, 'prompt_ms': 14000, 'cache_n': 0}, 'choices': [{'message': {'content': 'read same'}}]}
                (d / 'read36.json').write_text(json.dumps(j))
            else:
                (d / 'pipeline.trace').write_text(trace(ids[kind], arm))
                for cid in ids[kind]:
                    ms = 115 if arm == 'L' and kind == 'agent' else 120
                    (d / ('out_' + cid + '.json')).write_text(json.dumps(response(ms, cid + (' changed' if arm == 'L' else ''))))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(w.readout(root), 0, buf.getvalue())
        self.assertIn('LIVE_SWITCH GO', buf.getvalue())
        self.assertIn('STATE-CONFOUNDED', buf.getvalue())
        self.assertIn('STATE_WARNING NO_SHARED_STATE', buf.getvalue())
        result = json.loads((root / 'live_results.json').read_text())
        self.assertAlmostEqual(result['metrics']['agent/L']['median_pct'], 100 * (120 / 115 - 1))
        self.assertTrue(result['L0_identical'])
        p = root / 'live_L0_agent_p2/out_a0.json'
        p.write_text(json.dumps(response(text='different')))
        p = root / 'live_L_read36_p3/read36.json'
        j = json.loads(p.read_text()); j['timings']['cache_n'] = 100; p.write_text(json.dumps(j))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(w.readout(root), 1)
        result = json.loads((root / 'live_results.json').read_text())
        self.assertFalse(result['L0_identical'])
        self.assertTrue(any('fresh full read' in x for x in result['reasons']))
        print('CPU_LIVE_READER_FIXTURE', root)


if __name__ == '__main__':
    unittest.main()
