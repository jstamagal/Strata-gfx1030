"""CPU integration tests: real GGUF shards -> cache -> C++ loader/placement, corruption and STRP interoperability."""
import contextlib
import io
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gguf_reader as G
from iq_pack import Model
import q36_pack as P
import q36_profile as R

ROOT = Path(__file__).resolve().parent.parent


def string(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b


def fixture(root):
    # Deliberately not the real model shape. Quant types have different block sizes;
    # each layer's roles straddle shards and the last block stands in for MTP.
    ne, embd, ff, nl = 5, 512, 256, 3
    tensors = [[], []]
    for li, (gu, dt) in enumerate(((19, 22), (11, 12), (23, 8))):
        for ri, role in enumerate(('gate', 'up', 'down')):
            kind = gu if ri < 2 else dt
            be, bb = G.BLOCK_GEOMETRY[G.GGML_TYPES[kind]]
            n = ne * embd * ff // be * bb
            raw = bytes((i + 13 * li + 41 * ri) % 251 for i in range(n))
            shape = (embd, ff, ne) if ri < 2 else (ff, embd, ne)
            tensors[(li + ri) % 2].append((f'blk.{li}.ffn_{role}_exps.weight', shape, kind, raw))
    paths = []
    for si, rows in enumerate(tensors):
        path = root / f'test-{si+1:05}-of-00002.gguf'
        meta = {'split.no': si, 'split.count': 2, 'split.tensors.count': 9}
        if si == 0:
            meta.update({'general.architecture': 'qwen35moe', 'qwen35moe.block_count': nl,
                         'qwen35moe.expert_count': ne, 'qwen35moe.embedding_length': embd,
                         'qwen35moe.expert_feed_forward_length': ff, 'qwen35moe.nextn_predict_layers': 1})
        header = struct.pack('<IIQQ', G.GGUF_MAGIC, 3, len(rows), len(meta))
        for k, v in meta.items():
            header += string(k) + (struct.pack('<I', 8) + string(v) if isinstance(v, str)
                                   else struct.pack('<II', 4, v))
        payload = bytearray()
        for name, shape, kind, raw in rows:
            payload += bytes(-len(payload) % 32)
            header += string(name) + struct.pack('<I3QIQ', 3, *shape, kind, len(payload))
            payload += raw
        path.write_bytes(header + bytes(-len(header) % 32) + payload)
        paths.append(path)
    return paths


class PackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.exe = Path(cls.tmp.name) / 'io'
        subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', '-pthread', '-I', str(ROOT),
                        str(ROOT / 'tests/qwen35/q36_pack_io.cpp'), '-o', str(cls.exe)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_cache_payload_placement_and_rejections(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = fixture(root)
            m = Model(paths[0]); out = root / 'pack'
            with contextlib.redirect_stdout(io.StringIO()):
                P.build(m, out, tokenizer=False)
            shape, layers, total = P.layout(m)
            self.assertEqual((out / 'experts.bin').stat().st_size, total)
            # Independent source-slice oracle, including stride and tail zeros.
            with (out / 'experts.bin').open('r+b') as f:
                for li, layer in enumerate(layers):
                    for e in range(shape[1]):
                        f.seek(layer['offset'] + e * layer['stride'])
                        actual = f.read(layer['stride'])
                        pos = 0
                        for role in ('gate', 'up', 'down'):
                            name = f'blk.{li}.ffn_{role}_exps.weight'
                            g, t, _, path = m.where[name]
                            size = t.expected_bytes() // shape[1]
                            with path.open('rb') as source:
                                source.seek(g.data_start + t.offset + e * size)
                                self.assertEqual(actual[pos:pos+size], source.read(size))
                            pos += size
                        self.assertEqual(actual[pos:], bytes(len(actual)-pos))
                cmd = [str(self.exe), 'cache', str(out), *map(str, paths)]
                subprocess.run(cmd, check=True)
                before = paths[0].stat()
                os.utime(paths[0], (before.st_atime, before.st_mtime + 10))
                rejected = subprocess.run(cmd, capture_output=True)
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn(b'source identity mismatch', rejected.stderr)
                os.utime(paths[0], ns=(before.st_atime_ns, before.st_mtime_ns))
                # A wrong quant/layout byte is caught by independent builder verification.
                f.seek(layers[1]['offset'] + 7); f.write(b'\xff'); f.flush()
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    P.verify_layer(m, layers[1], shape[1], f)
                f.truncate(total-1)
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            with self.assertRaisesRegex(ValueError, 'exists'):
                P.build(m, out, tokenizer=False)

    def test_counts_profile_and_cpp_reader(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); trace = root/'counts'; profile = root/'profile'
            counts = [0]*15; counts[4]=7; counts[8]=3
            trace.write_bytes(b'Q36C'+struct.pack('<3I15Q',1,3,5,*counts))
            R.freeze([trace,trace],profile)
            subprocess.run([str(self.exe),'profile',str(profile),str(trace)],check=True)
            nl,ne,got=R.read_counts(trace)
            self.assertEqual((nl,ne),(3,5)); self.assertEqual(got[4],2); self.assertEqual(got[1],1)
            data=bytearray(profile.read_bytes()); data[28:32]=data[24:28]; profile.write_bytes(data)
            result=subprocess.run([str(self.exe),'profile',str(profile),str(trace)],capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn(b'duplicate',result.stderr)
            trace.write_bytes(trace.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError,'length'):
                R.read_counts(trace)


if __name__ == '__main__':
    unittest.main()
