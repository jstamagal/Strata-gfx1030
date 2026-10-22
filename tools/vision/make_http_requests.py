"""Write the OpenAI and Anthropic image requests for the manual curl smoke test."""
import argparse
import base64
import json
import mimetypes
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--config", type=Path, default=Path("vision-config.json"))
ap.add_argument("--image", type=Path, default=Path("build/vision-check-current/square.png"))
ap.add_argument("--out", type=Path, default=Path("build/vision-http"))
a = ap.parse_args()
cfg = json.loads(a.config.read_text())
mime = mimetypes.guess_type(a.image.name)[0]
assert mime in ("image/png", "image/jpeg", "image/webp", "image/gif")
data = base64.b64encode(a.image.read_bytes()).decode("ascii")
text = "Describe the shapes with their colors and quote all the text in the image."
base = {"model": cfg["model_name"], "max_tokens": 128, "temperature": 0, "stream": False}
openai = dict(base, messages=[{"role": "user", "content": [
    {"type": "text", "text": text},
    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}]}])
anthropic = dict(base, messages=[{"role": "user", "content": [
    {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}},
    {"type": "text", "text": text}]}])
a.out.mkdir(parents=True, exist_ok=True)
for name, request in [("openai", openai), ("anthropic", anthropic)]:
    path = a.out / (name + ".json")
    path.write_text(json.dumps(request, indent=2) + "\n")
    print(path)
