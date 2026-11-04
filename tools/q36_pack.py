"""Build a lossless q36 expert cache from all GGUF shards (CPU only).

Dense weights remain in the GGUF. The cache is [gate|up|down|zero stride padding]
per expert, with page-aligned layers and MMQ tail padding. Publication happens
only after independent per-layer SHA256 comparisons against the source slices.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import gguf_reader as G
from iq_pack import Model

ALIGN = 4096  # file layer alignment and q36's kMmqTail (not model dimensions)


def layout(model):
    meta = model.files[0].metadata
    arch = meta['general.architecture']
    if arch != 'qwen35moe':
        raise ValueError(f'expected qwen35moe, got {arch}')
    nl, ne, embd, ff = (int(meta[f'{arch}.{key}']) for key in
                        ('block_count', 'expert_count', 'embedding_length', 'expert_feed_forward_length'))
    if min(nl, ne, embd, ff) <= 0:
        raise ValueError('invalid model geometry')
    layers, offset = [], 0
    for li in range(nl):
        names = [f'blk.{li}.ffn_{role}_exps.weight' for role in ('gate', 'up', 'down')]
        ts = [model.where[name][1] for name in names]
        if [t.shape for t in ts] != [[embd, ff, ne], [embd, ff, ne], [ff, embd, ne]]:
            raise ValueError(f'layer {li}: unexpected expert tensor shapes')
        if ts[0].type_id != ts[1].type_id:
            raise ValueError(f'layer {li}: gate/up types differ')
        geom = [G.BLOCK_GEOMETRY[t.type_name] for t in ts]
        if any(t.shape[0] % g[0] for t, g in zip(ts, geom)):
            raise ValueError(f'layer {li}: partial quantization block in row')
        sizes = [t.expected_bytes() // ne for t in ts]
        al = math.lcm(geom[0][1], geom[2][1], 16)  # q36's existing MMQ stride rule
        size = sum(sizes)
        stride = (size + al - 1) // al * al
        offset = (offset + ALIGN - 1) // ALIGN * ALIGN
        layers.append(dict(layer=li, gu=ts[0].type_id, dt=ts[2].type_id, offset=offset,
                           stride=stride, up=sizes[0], down=sum(sizes[:2]), bytes=size,
                           names=names, sizes=sizes))
        offset += ne * stride + ALIGN
    return (nl, ne, embd, ff), layers, offset


def blob(model, layer, e):
    parts = [model.bytes(name)[e * size:(e + 1) * size].tobytes()
             for name, size in zip(layer['names'], layer['sizes'])]
    return b''.join(parts) + bytes(layer['stride'] - layer['bytes'])


def verify_layer(model, layer, ne, stream):
    source, cached = hashlib.sha256(), hashlib.sha256()
    stream.seek(layer['offset'])
    for e in range(ne):
        source.update(blob(model, layer, e))
        raw = stream.read(layer['stride'])
        if len(raw) != layer['stride']:
            raise ValueError(f"layer {layer['layer']}: truncated cache")
        cached.update(raw)
    if source.digest() != cached.digest() or stream.read(ALIGN) != bytes(ALIGN):
        raise ValueError(f"layer {layer['layer']}: cache/source checksum or MMQ tail mismatch")
    return source.hexdigest()


def build(model, out, tokenizer=True):
    shape, layers, total = layout(model)
    out.mkdir(parents=True, exist_ok=True)
    # Never replace a published cache while a process could be reading it.
    for name in ('q36_experts.txt', 'experts.bin', 'experts.bin.tmp'):
        if (out / name).exists():
            raise ValueError(f'{out / name} exists; use a new pack directory')
    sources = [(p.name, p.stat().st_size, int(p.stat().st_mtime)) for p in model.paths]
    start = time.monotonic()
    with (out / 'experts.bin.tmp').open('x+b') as f:
        for layer in layers:
            f.seek(layer['offset'])
            for e in range(shape[1]):
                f.write(blob(model, layer, e))
            f.write(bytes(ALIGN))
            f.flush()
            layer['sha256'] = verify_layer(model, layer, shape[1], f)
            print(f"layer {layer['layer']}: SHA256 {layer['sha256']} verified; "
                  f'{time.monotonic() - start:.1f}s', flush=True)
        f.flush()
        os.fsync(f.fileno())
    if sources != [(p.name, p.stat().st_size, int(p.stat().st_mtime)) for p in model.paths]:
        raise ValueError('source shards changed during build')
    if tokenizer:
        from strata_tokenizer import extract
        extract(model.paths[0], out)
    (out / 'experts.bin.tmp').rename(out / 'experts.bin')
    lines = ['Q36PACK 1 ' + ' '.join(map(str, (*shape, len(sources))))]
    lines += [f'{json.dumps(name)} {size} {mtime}' for name, size, mtime in sources]
    lines += [' '.join(str(l[k]) for k in
                       ('layer', 'gu', 'dt', 'offset', 'stride', 'up', 'down', 'bytes', 'sha256')) for l in layers]
    # Index is the completion marker, written last.
    (out / 'q36_experts.txt.tmp').write_text('\n'.join(lines) + '\n')
    (out / 'q36_experts.txt.tmp').rename(out / 'q36_experts.txt')
    print(f'published {out}: {total} bytes ({total / 2**30:.3f} GiB), {time.monotonic()-start:.1f}s')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gguf', required=True, type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--plan', action='store_true', help='read headers only; print exact cache size')
    ap.add_argument('--no-tokenizer', action='store_true')
    args = ap.parse_args()
    model = Model(args.gguf)
    shape, layers, total = layout(model)
    print(f'{shape[0]} blocks (including MTP), {shape[1]} experts, embd={shape[2]}, ff={shape[3]}')
    print(f'cache: {total} bytes = {total / 2**30:.3f} GiB; bounded scratch: one expert blob')
    print('Estimated disk traffic: twice source expert bytes + cache write + cache verification read.')
    if not args.plan:
        if not args.out:
            ap.error('--out is required without --plan')
        build(model, args.out, not args.no_tokenizer)


if __name__ == '__main__':
    main()
