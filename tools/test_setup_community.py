"""Tests for setup.py's community builds: bartowski's Swift 1.5 Q4_0 (three shards, the pack with --compat-bf16),
SC117's abliterated IQ3_S (the original model) and SC117's abliterated Swift 1.5 IQ3_XXS.  The published file names,
the pinned repositories with sizes and SHA-256, and setup.main() on a PC with two 32 GB gfx1030 cards (the AMD engine):
the downloads, the pack command, the config and the start script.  Mocked - no GPU, no downloads, nothing written
outside a temp folder.

    python -m unittest tools.test_setup_community
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

HUB = "https://huggingface.co/"
# family: (model, repository, pinned revision, folder of the shards in it, the published shard names)
BUILDS = {
    "bartowski-swift": ("Q4_0", "bartowski/ukisai_Swift-Qwen3.8-Flash-Next-GGUF",
                        "0a1c38334d2482040df430d35f9c22e4c898b492", "ukisai_Swift-Qwen3.8-Flash-Next-Q4_0/",
                        ["ukisai_Swift-Qwen3.8-Flash-Next-Q4_0-00001-of-00003.gguf",
                         "ukisai_Swift-Qwen3.8-Flash-Next-Q4_0-00002-of-00003.gguf",
                         "ukisai_Swift-Qwen3.8-Flash-Next-Q4_0-00003-of-00003.gguf"]),
    "sc117-abl": ("IQ3_S", "SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF",
                  "eec4e10a2c29b440a2ae85591fa11569ce752963", "IQ3_S/",
                  ["Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf",
                   "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf"]),
    "sc117-swift-abl": ("IQ3_XXS", "SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF",
                        "2b89cf3f7f2d9cb84275439e2c0cda83db15d32f", "IQ3_XXS/",
                        ["Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf",
                         "Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00002-of-00002.gguf"]),
}
# the files' bytes together, as the Hub lists them
BYTES = {"bartowski-swift": 100599253536, "sc117-abl": 83863586496, "sc117-swift-abl": 75827956000}
# the served name and the tag (pack folder, strata-<tag>.json, run-<tag>.sh)
NAMES = {"bartowski-swift": ("swift-qwen3.8-bartowski-q4_0", "bartowski-swift-q4_0"),
         "sc117-abl": ("qwen3.8-abl-iq3_s", "sc117-abl-iq3_s"),
         "sc117-swift-abl": ("swift-1.5-abliterated-iq3_xxs", "sc117-swift-abl-iq3_xxs")}


def v620(i):
    return {"index": i, "name": "AMD Radeon PRO V620", "vram_gb": 31.9, "arch": "gfx1030", "driver": "amdgpu",
            "vendor": "amd"}


class Tables(unittest.TestCase):
    def test_the_published_file_names(self):
        for family, (model, _, _, _, names) in BUILDS.items():
            fam = setup.FAMILIES[family]
            with self.subTest(family=family):
                self.assertEqual(setup.model_shards(fam, model), len(names))
                self.assertEqual([setup.model_file(fam, model, i) for i in range(1, len(names) + 1)], names)
                self.assertEqual(setup.gguf_choice(names[0]), (family, model))
                self.assertIsNone(setup.gguf_unsupported(names[0]))

    def test_the_pinned_repositories(self):
        for family, (model, repo, rev, folder, _) in BUILDS.items():
            fam = setup.FAMILIES[family]
            with self.subTest(family=family):
                self.assertEqual(setup.HF_REVISIONS[repo], rev)
                self.assertEqual(fam["hf"].format(q=model), f"{HUB}{repo}/resolve/{rev}/{folder}")
                self.assertEqual(fam["mmproj_hf"], f"{HUB}{repo}/resolve/{rev}/")

    def test_sizes_and_sha256(self):
        for family, (model, _, _, _, names) in BUILDS.items():
            fam = setup.FAMILIES[family]
            with self.subTest(family=family):
                self.assertEqual(sum(fam["sha256"][n][0] for n in names), BYTES[family])
                for n in names:
                    self.assertRegex(fam["sha256"][n][1], r"^[0-9a-f]{64}$")
                # the download the menu and the disk check count: the files' own size within 0.5 GB
                self.assertAlmostEqual(setup.MODELS[model]["download_gb"], BYTES[family] / 1e9, delta=0.5)

    def test_each_has_one_size_and_says_whose_it_is(self):
        for family, (model, _, _, _, _) in BUILDS.items():
            sizes = [m for m in setup.MODELS if family in setup.MODELS[m].get("families", ("qwen", "swift"))]
            self.assertEqual(sizes, [model], family)
            self.assertIn("community build", setup.FAMILIES[family]["by"])
        self.assertIn("bartowski", setup.MODELS["Q4_0"]["about"])
        self.assertIn("community build", setup.MODELS["Q4_0"]["about"])
        self.assertEqual(setup.MODELS["Q4_0"]["families"], ("bartowski-swift",))
        for family in ("sc117-abl", "sc117-swift-abl"):
            self.assertIn("abliterated", setup.FAMILIES[family]["by"])
            self.assertIn("SC117", setup.FAMILIES[family]["by"])

    def test_the_older_choices_are_as_they_were(self):
        self.assertEqual(list(setup.FAMILIES)[:4], ["qwen", "swift", "coder", "unsloth"])   # the menu's first four
        # upstream's order (0.2.0: UD-IQ4_XS first of Unsloth's, then UD-Q4_K_XL ... Q8_0); the community size last
        self.assertEqual(list(setup.MODELS)[:7], ["Q2_0", "IQ2_XS", "IQ3_XXS", "IQ3_S", "IQ1_M", "UD-IQ4_XS",
                                                  "UD-Q4_K_XL"])
        self.assertEqual(list(setup.MODELS)[-1], "Q4_0")
        for f, sizes in (("qwen", ["Q2_0", "IQ2_XS", "IQ3_XXS", "IQ3_S"]), ("swift", ["IQ2_XS", "IQ3_XXS", "IQ3_S"]),
                         ("coder", ["IQ1_M"]), ("unsloth", ["UD-IQ4_XS", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0"])):
            self.assertEqual([m for m in setup.MODELS if f in setup.MODELS[m].get("families", ("qwen", "swift"))],
                             sizes, f)
        for m, gb in (("IQ3_XXS", (75.8, 60, 42.9)), ("IQ3_S", (83.6, 62, 50.3))):
            self.assertEqual(tuple(setup.MODELS[m][k] for k in ("download_gb", "ram_gb", "arena_gb")), gb)

    def test_sc117_image_encoder_is_not_the_originals_file(self):
        """SC117's encoder has the original's file name and other bytes: a folder of its own, shared by its two
        builds (the same SHA-256 in both repositories)."""
        a, b = setup.FAMILIES["sc117-abl"], setup.FAMILIES["sc117-swift-abl"]
        self.assertEqual(a["mmproj"], setup.FAMILIES["qwen"]["mmproj"])
        self.assertEqual((a["mmproj_dir"], b["mmproj_dir"]), ("sc117", "sc117"))
        self.assertEqual(a["mmproj"], b["mmproj"])
        self.assertNotIn("mmproj_dir", setup.FAMILIES["qwen"])

    def test_choices_from_config(self):
        with tempfile.TemporaryDirectory() as t:
            for family, (model, *_) in BUILDS.items():
                p = Path(t) / f"strata-{NAMES[family][1]}.json"
                p.write_text(json.dumps({"args": ["--max-context", "8192"]}))
                ch = setup.choices_from_config(p)
                self.assertEqual((ch["family"], ch["model"]), (family, model), p.name)


class FakeGGUF:
    """gguf_reader.GGUFFile: the PLE table is in shard 2 of the original's layout, in shard 1 of Swift 1.5's."""
    def __init__(self, path):
        name = Path(path).name
        ple = "00001-of" if name.startswith(("Swift-", "ukisai_Swift-")) else "00002-of"
        self.tensors = [types.SimpleNamespace(name="per_layer_token_embd.weight" if ple in name
                                              else "blk.0.ffn_up_exps.weight")]


class Install(unittest.TestCase):
    """setup.main() --yes --backend hip --gpus 0,1 on 2x gfx1030 with 92 GB of RAM, every outside effect mocked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        (self.t / "data" / "mtp" / "rt").mkdir(parents=True)
        (self.t / "data" / "mtp" / "rt" / "experts.bin").write_bytes(b"")
        (self.t / "serve").mkdir()
        self.downloads, self.runs, self.verified = [], [], []

    def tearDown(self):
        self.tmp.cleanup()

    def main(self, family, argv=(), ram=92.0, before=()):
        """before: files already in the models folder, (relative path, finished)."""
        model = BUILDS[family][0]
        eng = self.t / "engine"
        eng.mkdir(exist_ok=True)
        (eng / "BUILD.json").write_text(json.dumps({"version": "0.1.43.1", "source": "local", "lib_dirs": ["<rocm>"]}))
        (eng / "strata-vision").write_bytes(b"x")       # the CPU picture reader (strata_control/vision_reader.py)
        for rel in before:
            f = self.t / "models" / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b"")
            setup.mark(f)

        def fake_download(url, dst, what=None):
            self.downloads.append((url, dst))
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"")
            setup.mark(dst)

        patches = [
            mock.patch.object(setup, "ROOT", self.t),
            mock.patch.object(setup, "WIN", False),
            mock.patch.object(setup, "EXE", "strata"),
            mock.patch.object(setup, "VEXE", "strata-vision"),
            mock.patch.object(setup, "GPU_PICK", None),
            mock.patch.object(setup, "data_folder", lambda d, *a, **k: (self.t / "data", [])),
            mock.patch.object(setup, "load_settings", lambda: {}),
            mock.patch.object(setup, "save_settings", lambda s: None),
            mock.patch.object(setup, "gpus", lambda: []),
            mock.patch.object(setup, "amd_gpus", lambda *a: [v620(0), v620(1)]),
            mock.patch.object(setup, "amd_pci_devices", lambda *a: []),
            mock.patch.object(setup, "amd_device_access_problem", lambda *a, **k: None),
            mock.patch.object(setup, "linux_desktop", lambda *a: False),
            mock.patch.object(setup, "ram_gb", lambda: ram),
            mock.patch.object(setup, "cpu_info", lambda: ("Test CPU", True, True)),
            mock.patch.object(setup, "cpu_cores", lambda: None),
            mock.patch.object(setup, "page_file_gb", lambda: 16.0),
            mock.patch.object(setup, "is_wsl", lambda: False),
            mock.patch.object(setup, "rotational_disk", lambda p: None),
            mock.patch.object(setup, "free_gb", lambda p: 900.0),
            mock.patch.object(setup, "pip_install", lambda *a, **k: None),
            mock.patch.object(setup, "get_llama_cpp", lambda: self.t / "llama.cpp"),
            mock.patch.object(setup, "build_engine_hip", lambda *a, **k: eng),
            mock.patch.object(setup, "get_prebuilt", mock.Mock(side_effect=AssertionError("the NVIDIA engine"))),
            mock.patch.object(setup, "update_installed_engine", lambda *a, **k: None),
            mock.patch.object(setup, "download", fake_download),
            mock.patch.object(setup, "check_shards", lambda shards: None),
            mock.patch.object(setup, "verify_sha256", lambda s, size, sha, *a, **k: self.verified.append((s.name, size, sha))),
            mock.patch.object(setup, "run", lambda cmd, *a, **k: self.runs.append([str(x) for x in cmd])),
            mock.patch.object(setup, "mtp_corrupt", lambda *a, **k: False),
            mock.patch.object(setup, "refresh_draft_vocab", lambda *a, **k: None),
            mock.patch.object(setup, "saved_calibration", lambda cfg: None),
            mock.patch.object(setup, "calibrate_config", mock.Mock(side_effect=AssertionError("calibrated"))),
            mock.patch.object(setup, "start", mock.Mock(side_effect=AssertionError("started"))),
            mock.patch.dict(sys.modules, {"gguf_reader": types.SimpleNamespace(GGUFFile=FakeGGUF)}),
            mock.patch.object(sys, "argv", ["setup.py", "--yes", "--no-start", "--backend", "hip", "--gpus", "0,1",
                                            "--family", family, "--model", model, "--context", "131072",
                                            "--models-dir", str(self.t / "models"), *argv]),
            mock.patch("builtins.input", mock.Mock(side_effect=AssertionError("asked with --yes"))),
        ]
        out = io.StringIO()
        code = None
        with contextlib.ExitStack() as st:
            for p in patches:
                st.enter_context(p)
            with contextlib.redirect_stdout(out):
                try:
                    code = setup.main()
                except SystemExit as e:
                    code = e.code
        cfg = self.t / f"strata-{NAMES[family][1]}.json"
        return code, out.getvalue(), (json.loads(cfg.read_text()) if cfg.exists() else None)

    def check(self, family, cfg, out, vision=None):
        """What every build's config says on the two cards."""
        model, repo, rev, folder, names = BUILDS[family]
        served, tag = NAMES[family]
        models, args = self.t / "models", cfg["args"]
        val = lambda k: args[args.index(k) + 1]   # noqa: E731
        self.assertEqual(val("--native"), str(models / (setup.FAMILIES[family]["tag"] + model) / names[0]))
        self.assertEqual(val("--pack"), str(self.t / "data" / "packs" / tag))
        self.assertEqual(cfg["tokenizer"], str(self.t / "data" / "packs" / tag / "tokenizer"))
        self.assertEqual(val("--max-context"), "131072")
        self.assertEqual(cfg["model_name"], served)
        self.assertEqual(cfg["log"], str(self.t / f"strata-{tag}.log"))
        self.assertEqual(cfg["backend"], "hip")
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertTrue(cfg["gpus_asked"])
        self.assertEqual(cfg["layer_split"], "auto")
        self.assertLessEqual(set(cfg.get("env", {})), setup.SETUP_ENV)   # the RDNA2 switches: the engine's defaults
        for flag in ("--resident-budget-gib", "--mmap-experts", "--resident-experts", "--control-vector-scaled"):
            self.assertNotIn(flag, args)                    # all of its experts in RAM, no speed projection
        # the shards: each from its pinned folder, each checked against its pinned size and SHA-256
        base = f"{HUB}{repo}/resolve/{rev}/"
        want = [(base + folder + n, models / (setup.FAMILIES[family]["tag"] + model) / n) for n in names]
        self.assertEqual([d for d in self.downloads if "mmproj" not in d[0]], want)
        self.assertEqual(self.verified, [(n, *setup.COMMUNITY_SHARDS[n]) for n in names])
        packs = [r for r in self.runs if r[1].endswith("iq_pack.py")]
        self.assertEqual(len(packs), 1, self.runs)
        self.assertEqual(packs[0][packs[0].index("--gguf") + 1], val("--native"))
        self.assertEqual(packs[0][packs[0].index("--out") + 1], val("--pack"))
        self.assertNotIn("--experts-bin", packs[0])
        script = self.t / f"run-{tag}.sh"
        self.assertTrue(script.is_file(), out)
        self.assertIn(f'"--config" "{self.t / f"strata-{tag}.json"}"', script.read_text())
        return val, packs[0]

    def test_bartowski_swift_q4_0(self):
        code, out, cfg = self.main("bartowski-swift")
        self.assertEqual(code, 0, out)
        val, pack = self.check("bartowski-swift", cfg, out)
        self.assertIn("--compat-bf16", pack)                # its small projections are Q4_0: read as BF16
        self.assertNotIn("--ple-gguf", cfg["args"])         # three shards: the engine finds the PLE table's itself
        self.assertIn("vision", cfg)                        # --yes: images on (upstream 0.2.0: wherever the model reads them)
        self.assertIn("Swift Open License 1.0", out)
        self.assertIn("size: Q4_0", out)
        self.assertNotIn("low-RAM mode", out)               # 92 GB: all 68 GB of experts in RAM

    def test_bartowski_swift_q4_0_images(self):
        code, out, cfg = self.main("bartowski-swift", ["--vision", "cpu"])
        self.assertEqual(code, 0, out)
        mm = self.t / "models" / "mmproj-ukisai_Swift-Qwen3.8-Flash-Next-f16.gguf"
        rev = BUILDS["bartowski-swift"][2]
        self.assertIn((f"{HUB}bartowski/ukisai_Swift-Qwen3.8-Flash-Next-GGUF/resolve/{rev}/{mm.name}", mm),
                      self.downloads)
        self.assertEqual(cfg["vision"]["mmproj"], str(mm))
        self.assertFalse(cfg["vision"]["gpu"])
        self.assertTrue(cfg["vision"]["model"].endswith("Q4_0-00001-of-00003.gguf"))

    def test_sc117_abl_iq3_s(self):
        code, out, cfg = self.main("sc117-abl")
        self.assertEqual(code, 0, out)
        val, pack = self.check("sc117-abl", cfg, out)
        self.assertNotIn("--compat-bf16", pack)             # GSQ-RCO files: packed as the original's
        self.assertTrue(val("--ple-gguf").endswith("abliterated-IQ3_S-00002-of-00002.gguf"))   # the original's layout
        self.assertNotIn("Its license", out)

    def test_sc117_swift_abl_iq3_xxs(self):
        code, out, cfg = self.main("sc117-swift-abl")
        self.assertEqual(code, 0, out)
        val, pack = self.check("sc117-swift-abl", cfg, out)
        self.assertNotIn("--compat-bf16", pack)
        self.assertEqual(val("--ple-gguf"), val("--native"))    # Swift 1.5's layout: the PLE table in shard 1
        self.assertIn("Swift Open License 1.0", out)

    def test_sc117_image_encoder_beside_the_originals(self):
        """The original's encoder is already in the models folder under the same name: SC117's is downloaded into
        its own folder and the config names that one."""
        for family in ("sc117-abl", "sc117-swift-abl"):
            self.downloads, self.runs, self.verified = [], [], []
            code, out, cfg = self.main(family, ["--vision", "cpu"], before=["mmproj-Qwen3.8-Flash-Next-BF16.gguf"])
            self.assertEqual(code, 0, out)
            _, repo, rev, _, _ = BUILDS[family]
            mm = self.t / "models" / "sc117" / "mmproj-Qwen3.8-Flash-Next-BF16.gguf"
            self.assertEqual(cfg["vision"]["mmproj"], str(mm), family)
            if family == "sc117-abl":                       # the second build finds the first one's copy
                self.assertIn((f"{HUB}{repo}/resolve/{rev}/{mm.name}", mm), self.downloads)

    def test_sc117_abl_shares_the_originals_shard_2(self):
        """SC117's shard 2 is ISTA-DASLab's file (the same SHA-256): one that is already here is linked, not
        downloaded again, and still checked against the pinned hash."""
        code, out, cfg = self.main("sc117-abl", before=["IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"])
        self.assertEqual(code, 0, out)
        names = BUILDS["sc117-abl"][4]
        self.assertEqual([d[1].name for d in self.downloads if "mmproj" not in d[0]], names[:1])
        self.assertIn("shared with IQ3_S (identical file)", out)
        self.assertEqual([v[0] for v in self.verified], names)

    def test_swift_builds_do_not_share_it(self):
        code, out, cfg = self.main("sc117-swift-abl",
                                   before=["IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"])
        self.assertEqual(code, 0, out)
        self.assertEqual([d[1].name for d in self.downloads if "mmproj" not in d[0]], BUILDS["sc117-swift-abl"][4])

    def test_q4_0_on_64_gb_is_the_low_ram_mode(self):
        """Not measured with this file: what setup decides for it is what it decides for any size whose experts do
        not fit the RAM (the experts written into one file for the low-RAM mode)."""
        self.assertTrue(setup.low_ram_needed("Q4_0", 63.7))
        self.assertFalse(setup.low_ram_needed("Q4_0", 92.0))
        code, out, cfg = self.main("bartowski-swift", ram=63.7)
        self.assertEqual(code, 0, out)
        packs = [r for r in self.runs if r[1].endswith("iq_pack.py")]
        self.assertEqual(["--experts-bin" in r for r in packs], [False, True])


if __name__ == "__main__":
    unittest.main()
