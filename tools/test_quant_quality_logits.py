#!/usr/bin/env python3
"""CPU-only checks for the frozen-prompt raw-logit quality comparator."""
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).with_name("quant_quality.py")


class RawLogits(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def vector(self, name, xs):
        path = self.root / name
        path.write_bytes(struct.pack("<" + "f" * len(xs), *xs))
        return path

    def run_compare(self, ref, run, *extra):
        return subprocess.run([sys.executable, "-I", str(TOOL), "logits", str(ref), str(run), *extra],
                              text=True, capture_output=True)

    def test_identical_and_ties(self):
        p = self.vector("ref", [1, 1, -2])
        r = self.run_compare(p, p, "--topk", "2")
        self.assertEqual(r.returncode, 0, r.stderr)
        d = json.loads(r.stdout)
        self.assertTrue(d["same_bytes"])
        self.assertEqual(d["reference_top_ids"], [0, 1])
        self.assertEqual(d["kl_ref_to_run"], 0)
        self.assertEqual(d["max_abs_logit_error"], 0)

    def test_known_kl_and_topk(self):
        p = self.vector("ref", [0, math.log(3)])
        q = self.vector("run", [0, 0])
        out = self.root / "metrics.json"
        r = self.run_compare(p, q, "--topk", "1", "--json", str(out))
        self.assertEqual(r.returncode, 0, r.stderr)
        d = json.loads(r.stdout)
        self.assertAlmostEqual(d["kl_ref_to_run"], .25*math.log(.5)+.75*math.log(1.5), places=7)
        self.assertFalse(d["top1_agrees"])
        self.assertEqual(d["topk_overlap"], 0)
        self.assertEqual(json.loads(out.read_text())["runs"][str(q)]["vocab"], 2)

    def test_large_logits_do_not_overflow(self):
        p = self.vector("ref", [10000, 10001, -10000])
        q = self.vector("run", [10001, 10002, -9999])
        r = self.run_compare(p, q)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertLess(abs(json.loads(r.stdout)["kl_ref_to_run"]), 1e-10)

    def test_rejects_wrong_shape_nonfinite_and_short(self):
        p = self.vector("ref", [0, 1])
        for i, xs in enumerate(([0], [math.nan, 1], [math.inf, 0], [])):
            with self.subTest(xs=xs):
                q = self.vector(f"bad{i}", xs)
                self.assertNotEqual(self.run_compare(p, q).returncode, 0)
        q = self.root / "short"
        q.write_bytes(b"123")
        self.assertNotEqual(self.run_compare(p, q).returncode, 0)
        self.assertNotEqual(self.run_compare(p, p, "--topk", "0").returncode, 0)


if __name__ == "__main__":
    unittest.main()
