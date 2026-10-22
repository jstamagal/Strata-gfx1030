"""Compare strata-vision's ENC rows with an independently built llama.cpp mtmd.

No text weights or GPUs are used. Writes inputs, both SVE files, logs and report.json
under --out. The reference also checks every mtmd decoder position against SVE order.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

import numpy as np
from PIL import Image, ImageDraw


def read_sve(path):
    data = path.read_bytes()
    magic, n, nx, ny, width = struct.unpack("<5i", data[:20])
    assert magic == 0x31455653 and n == nx * ny and width > 0
    rows = np.frombuffer(data, dtype="<f4", offset=20).reshape(n, width)
    assert np.isfinite(rows).all(), f"nonfinite embedding: {path}"
    return (n, nx, ny, width), rows.astype(np.float64)


def fixtures(out):
    result = []
    for name, size in [("square", (224, 224)), ("wide", (448, 224)), ("tall", (224, 448))]:
        image = Image.new("RGB", size, "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((12, 12, 92, 92), fill="red")
        draw.ellipse((112, 12, 192, 92), fill="blue")
        draw.text((24, 120), "ORNITH 4217", fill="black", font_size=24)
        path = out / (name + ".png")
        image.save(path)
        result.append(path)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vision", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--image", type=Path, action="append")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--min-tokens", type=int, default=64)
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--rtol", type=float, default=1e-6)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    images = args.image or fixtures(out)
    commands, comparisons = [], []
    for i, image in enumerate(images):
        image = image.resolve()
        sve, ref = out / f"{i}.strata.sve", out / f"{i}.mtmd.sve"
        cmd = [str(args.vision.resolve()), "--model", args.model, "--mmproj", args.mmproj,
               "--native-f16", "--threads", str(args.threads), "--max-tokens", str(args.max_tokens),
               "--min-tokens", str(args.min_tokens)]
        commands.append(cmd)
        with (out / f"{i}.strata.log").open("w") as log:
            run = subprocess.run(cmd, input=f"ENC {image} {sve}\nQUIT\n", text=True,
                                 stdout=subprocess.PIPE, stderr=log, timeout=600, check=True)
        (out / f"{i}.protocol.log").write_text(run.stdout)
        assert run.stdout.startswith("READY ") and "\nOK " in run.stdout, run.stdout
        cmd = [str(args.reference.resolve()), args.model, args.mmproj, str(image), str(ref),
               str(args.threads), str(args.max_tokens), str(args.min_tokens)]
        commands.append(cmd)
        with (out / f"{i}.mtmd.log").open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=log, timeout=600, check=True)
        shape, actual = read_sve(sve)
        ref_shape, expected = read_sve(ref)
        assert shape == ref_shape, (shape, ref_shape)
        diff = np.abs(actual - expected)
        row = {"image": str(image), "shape_n_nx_ny_width": shape,
               "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
               "strata_sha256": hashlib.sha256(sve.read_bytes()).hexdigest(),
               "reference_sha256": hashlib.sha256(ref.read_bytes()).hexdigest(),
               "max_abs": float(diff.max()),
               "max_relative_floor_1e-8": float((diff / np.maximum(np.abs(expected), 1e-8)).max()),
               "relative_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(expected), 1e-30)),
               "bit_identical": sve.read_bytes() == ref.read_bytes(),
               "passed": bool(np.allclose(actual, expected, atol=args.atol, rtol=args.rtol))}
        comparisons.append(row)
        print(json.dumps(row), flush=True)
    report = {"comparisons": comparisons, "commands": commands,
              "atol": args.atol, "rtol": args.rtol}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not all(row["passed"] for row in comparisons):
        raise SystemExit("embedding comparison failed")


if __name__ == "__main__":
    main()
