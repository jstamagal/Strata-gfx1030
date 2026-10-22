"""GPU regression for TP vision, graph replay, allocation growth and MTP.

By default only prepares files. --run starts the engine's stdin protocol (no HTTP
listener), so run it only when both GPUs are available. Uses the small images from
check_embeddings.py, allowing --prefill 4 to exercise prompt graphs within images.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strata_tokenizer import Tokenizer


def prepare(cfg, fixtures, out):
    pack = Path(cfg["tokenizer"])
    vocab = json.loads((pack / "vocab.json").read_text())
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    tok = Tokenizer(tokens, (pack / "merges.txt").read_text().split("\n"),
                    json.loads((pack / "token_type.json").read_text()))
    enc = lambda s: tok.encode(s, parse_special=True)
    pad = vocab.get("<|image_pad|>")
    assert pad is not None
    requests = []
    # A growing d_img must not invalidate graphs; equal-size different images and
    # an intervening text request must not retain stale rows, tables or offsets.
    for tag, indices in [("small", [0]), ("grown", [1, 2]), ("text", []),
                         ("tall", [2]), ("wide", [1]), ("small_again", [0])]:
        ids = enc("<|im_start|>user\n")
        records = []
        for index in indices:
            data = (fixtures / f"{index}.strata.sve").read_bytes()
            magic, n, nx, ny, width = struct.unpack("<5i", data[:20])
            assert magic == 0x31455653 and n == nx * ny and len(data) == 20 + 4 * n * width
            ids += enc("<|vision_start|>") + [pad] * n + enc("<|vision_end|>\n")
            records.append(data)
        ids += enc("Describe the shapes and quote the text." if indices else "What is two plus two?")
        ids += enc("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        emb = out / (tag + ".sve")
        if records:
            emb.write_bytes(b"".join(records))
        (out / (tag + ".ids")).write_text(",".join(map(str, ids)) + "\n")
        requests.append((tag, (f"GENI 24 temperature=0 {emb}" if records else "GEN 24 temperature=0")
                         + " " + ",".join(map(str, ids))))
    (out / "requests.txt").write_text("\n".join(line for _, line in requests) + "\n")
    return requests


def run_case(name, cfg, out, requests, mtp, graph, vision=True, verify=False):
    env = dict(os.environ, **cfg["env"])
    env.update(STRATA_Q36_TP="1", STRATA_Q36_TP_VISION_GRAPH=str(int(graph)), STRATA_Q36_TP_CHECK="1")
    # The existing VERIFY switch checks presence, so even "0" would disable graphs.
    env.pop("STRATA_Q36_VERIFY", None)
    if verify:
        env["STRATA_Q36_VERIFY"] = "1"
    env["LD_LIBRARY_PATH"] = ":".join(cfg.get("lib_dirs", []) + [env.get("LD_LIBRARY_PATH", "")])
    cmd = [str(Path(cfg["cwd"]) / "build/strata-q36"), "--model", cfg["vision"]["model"],
           "--max-context", "4096", "--prefill", "4", "--mtp", str(mtp), "--serve"]
    if vision:
        cmd.append("--vision")
    if verify:
        cmd.append("--no-graph")
    with (out / (name + ".stderr")).open("w") as log:
        result = subprocess.run(cmd, input="\n".join(line for _, line in requests) + "\n",
                                text=True, stdout=subprocess.PIPE, stderr=log,
                                env=env, timeout=3600, check=True)
    (out / (name + ".stdout")).write_text(result.stdout)
    answers, current = [], []
    for line in result.stdout.splitlines():
        if line.startswith("ERR "):
            raise RuntimeError(line)
        if line.startswith("T "):
            current.append(int(line.split()[1]))
        if line.startswith("DONE "):
            assert current, f"empty answer: {name}"
            answers.append(current)
            current = []
    assert len(answers) == len(requests), f"missing replies: {name}"
    log = (out / (name + ".stderr")).read_text()
    if verify:
        assert "VERIFY MISMATCH" not in log, f"kernel mismatch: {out}/{name}.stderr"
    if graph and vision:
        assert "TP_VISION_GRAPH captured main" in log, "image graph was not exercised"
        if mtp:
            assert "TP_VISION_GRAPH captured MTP" in log, "MTP image graph was not exercised"
    return {"command": cmd, "answers": answers}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("vision-config.json"))
    ap.add_argument("--fixtures", type=Path, default=Path("build/vision-check-current"))
    ap.add_argument("--out", type=Path, default=Path("build/vision-tp-check"))
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--verify", action="store_true", help="run STRATA_Q36_VERIFY=1 kernel comparisons")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    requests = prepare(cfg, args.fixtures.resolve(), out)
    if not args.run:
        print(f"Prepared {out}/requests.txt; pass --run when the GPUs are available.")
        return
    report = {}
    if args.verify:
        report["verify"] = run_case("verify", cfg, out, requests, 3, False, verify=True)
    else:
        for mtp in (0, 3):
            for graph in (False, True):
                name = f"mtp{mtp}_imagegraph{int(graph)}"
                report[name] = run_case(name, cfg, out, requests, mtp, graph)
        text_requests = [(tag, line) for tag, line in requests if tag == "text"]
        report["text_default"] = run_case("text_default", cfg, out, text_requests, 0, False, vision=False)
        expected = report["mtp0_imagegraph0"]["answers"]
        for name, result in report.items():
            result["matches"] = result["answers"] == ([expected[2]] if name == "text_default" else expected)
        # Allocation growth / reuse must also preserve the first image's answer.
        report["repeat_matches"] = all(result["answers"][0] == result["answers"][-1]
                                       for name, result in report.items() if name != "text_default")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not args.verify:
        assert report["repeat_matches"] and all(v["matches"] for v in report.values() if isinstance(v, dict)), \
            f"token mismatch; inspect {out}/report.json and logs (not a measured quality pass)"
    print(f"Passed; results in {out}/report.json")


if __name__ == "__main__":
    main()
