#!/usr/bin/env python3
"""Live candidate: host checks now; configure, HIP compile and GPU parity ONLY in the live build slot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

W = Path(__file__).resolve().parents[2]
R = Path('/home/ronald_rump/Strata_donttouch')
H = Path('/home/ronald_rump/inbox/agents/claude/strata-ab')
S, B = W / 'scratch', W / 'build-hip'
TRASH = Path('/home/ronald_rump/inbox/agents/claude/trash/2026-11-05')
PACK = Path('/home/ronald_rump/Strata-data/packs/sc117-abl-iq3_s')
REQUIRED = ('c6ba587e9', '6738da922', '806ecab63', 'bf07bf391')
TARGETS = ('strata', 'hc_i8_bench', 'hip_t1_diet_parity', 'hip_route_plan_parity',
           'hip_qsa_fuse_parity', 'hip_node_fuse_parity', 'hip_mtp_fuse_parity', 'gr_parity', 'verify_parity')
L_ENV = {'STRATA_HC_REQ8': '1', 'STRATA_HC_I8_ROWS': '4,2,2,2,2,2,2,2',
         'STRATA_HC_I8_UPR': '2,2,2,1,1,1,1,1', 'STRATA_T1_DIET': '12', 'STRATA_PL_POLICY_CLIP': '2.10'}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(W), *args], text=True).strip()


def command(args, **kw):
    print('+', ' '.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kw)


def slot():
    if not (H / 'buildslot.open').is_file() or 'live' not in (H / 'buildslot.expect').read_text().split():
        raise RuntimeError('GATE FAIL: live build slot is not open; NO configure/HIP build/GPU execution allowed')


def source_gate():
    if git('branch', '--show-current') != 'basalt38/live-1105':
        raise RuntimeError('GATE FAIL: wrong branch')
    for c in REQUIRED:
        command(['git', '-C', W, 'merge-base', '--is-ancestor', c, 'HEAD'])
    if subprocess.run(['git', '-C', str(W), 'merge-base', '--is-ancestor', '31398223c', 'HEAD']).returncode == 0:
        raise RuntimeError('GATE FAIL: native/q4n MTP branch must NOT be merged')
    drift = git('log', '--oneline', 'HEAD..' + git('rev-parse', 'main'), '--',
                'src', 'include', 'cmake', 'CMakeLists.txt', 'serve', 'third_party')
    if drift:
        raise RuntimeError('GATE FAIL: main engine drift: ' + drift)
    command(['git', '-C', W, 'diff', '--quiet'])
    command(['git', '-C', W, 'diff', '--cached', '--quiet'])
    print('SOURCE_GATE PASS', git('rev-parse', 'HEAD'), flush=True)


def options(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        m = re.fullmatch(r'([^:#]+):([^=]+)=(.*)', line)
        if not m or m[2] in ('INTERNAL', 'STATIC'):
            continue
        k, t, v = m.groups()
        if (k.startswith(('STRATA_', 'GGML_')) or k in ('BUILD_SHARED_LIBS', 'GPU_TARGETS', 'CMAKE_BUILD_TYPE',
                'CMAKE_PREFIX_PATH', 'CMAKE_HIP_ARCHITECTURES', 'CMAKE_HIP_COMPILER_ROCM_ROOT', 'CMAKE_HIP_PLATFORM')
                or re.fullmatch(r'CMAKE_(C|CXX|HIP|ASM)_COMPILER', k)
                or re.fullmatch(r'CMAKE_(C|CXX|HIP|ASM|EXE_LINKER|SHARED_LINKER|MODULE_LINKER|STATIC_LINKER)_FLAGS(_.*)?', k)):
            result[k] = (t, v)
    return result


def inputs():
    vendor = R / 'third_party/llama.cpp/ggml/src/ggml-cuda'
    files = [R / 'build-hip/strata', R / 'build-hip/CMakeCache.txt', R / 'strata-sc117-IQ3_S-qwen-abliterated.json']
    files += [vendor / name for name in ('mmq.cuh', 'mmq-config-rdna2.cuh', 'common.cuh', 'vecdotq.cuh', 'quantize.cu')]
    return {str(p): sha(p) for p in files}


def archive(path):
    path = Path(path)
    if path.exists():
        TRASH.mkdir(parents=True, exist_ok=True)
        dst = TRASH / (path.name + '.live-' + str(time.time_ns()))
        path.rename(dst)
        print('ARCHIVED', dst, flush=True)


def host_checks(out=None):
    S.mkdir(parents=True, exist_ok=True)
    out = out or S / ('host-checks-' + str(time.time_ns()))
    out.mkdir(parents=True, exist_ok=True)
    exe = out / 'draft_policy_test'
    command(['nice', '-n', '19', 'ionice', '-c3', '/usr/bin/c++', '-std=c++17', '-O2', '-I' + str(W / 'include'),
             W / 'src/spec/draft_policy.cpp', W / 'src/spec/draft_policy_test.cpp', '-o', exe])
    with (out / 'draft_policy_test.out').open('w') as f:
        command([exe], stdout=f, stderr=subprocess.STDOUT)
    text = (out / 'draft_policy_test.out').read_text()
    if not text.rstrip().endswith('PASS'):
        raise RuntimeError('GATE FAIL: policy CPU verdict missing')
    print(text, flush=True)
    with (out / 'bimodal-reader-test.out').open('w') as f:
        command(['python3', '-I', W / 'tools/hip/bimodal-read-test.py', H / 'ab', out / 'reader-fixtures'],
                stdout=f, stderr=subprocess.STDOUT)
    print((out / 'bimodal-reader-test.out').read_text(), flush=True)
    print('HOST_POLICY_CHECKS PASS', out, flush=True)
    return out


def verified_build():
    source_gate()
    record = json.loads((H / 'staging/live-build.json').read_text())
    if record['commit'] != git('rev-parse', 'HEAD') or record['inputs'] != inputs():
        raise RuntimeError('GATE FAIL: stale live build or main/config/vendor changed')
    if sha(B / 'CMakeCache.txt') != record['cache']:
        raise RuntimeError('GATE FAIL: live cache changed')
    for p, h in record['binaries'].items():
        if sha(p) != h:
            raise RuntimeError('GATE FAIL: binary changed: ' + p)
    if (H / 'staging/live-build.log').read_text().splitlines()[-1] != 'build exit 0':
        raise RuntimeError('GATE FAIL: live build/parity did not finish successfully')
    return record


def build():
    slot()
    source_gate()
    before, head = inputs(), git('rev-parse', 'HEAD')
    for name in ('live-build.json', 'live-build.sha256', 'live-build.commit'):
        archive(H / 'staging' / name)
    out = S / 'live-tests'
    archive(out)
    out.mkdir(parents=True)
    expected = options(R / 'build-hip/CMakeCache.txt')
    if expected['STRATA_GGML_DIR'][1] != str(R / 'third_party/llama.cpp'):
        raise RuntimeError('GATE FAIL: main no longer uses its unmodified vendor source')
    # No private vendor snapshot, no patch application, no cache overrides.
    flags = [f'-D{k}{":" + t if t != "UNINITIALIZED" else ""}={v}' for k, (t, v) in expected.items()]
    env = os.environ | {'HIP_VISIBLE_DEVICES': '-1', 'ROCR_VISIBLE_DEVICES': '-1'}
    slot()
    command(['nice', '-n', '19', 'cmake', '-S', W, '-B', B, '-G', 'Ninja', *flags], env=env)
    actual = options(B / 'CMakeCache.txt')
    diff = [f'{k}: main={expected.get(k)} live={actual.get(k)}' for k in expected.keys() | actual.keys()
            if (expected.get(k) or (None, None))[1] != (actual.get(k) or (None, None))[1]]
    if diff:
        raise RuntimeError('GATE FAIL: exact cache replay differs: ' + '; '.join(diff))
    print('CACHE_GATE PASS main/live: ALL options match, including the SAME unmodified vendor path', flush=True)
    slot()
    command(['nice', '-n', '19', 'cmake', '--build', B, '-j16', '--target', *TARGETS], env=env)
    host_checks(out / 'policy-cpu')
    live = json.loads((R / 'strata-sc117-IQ3_S-qwen-abliterated.json').read_text())
    clean = {k: v for k, v in os.environ.items() if not k.startswith(('STRATA_', 'HIP_VISIBLE', 'ROCR_VISIBLE', 'CUDA_VISIBLE', 'GPU_DEVICE_ORDINAL'))}
    clean.update({k: str(v) for k, v in live['env'].items()})
    clean.update(STRATA_HC_REQ8='0', STRATA_HC_Q8='0', STRATA_T1_DIET='0')
    def test(name, args, extra=None):
        slot()
        with (out / (name + '.out')).open('w') as f:
            p = subprocess.run(list(map(str, args)), env=clean | {'HIP_VISIBLE_DEVICES': '0'} | (extra or {}),
                               stdout=f, stderr=subprocess.STDOUT, timeout=1200)
        print('TEST_STATUS', name, p.returncode, 'log', out / (name + '.out'), flush=True)
        text = (out / (name + '.out')).read_text()
        if p.returncode:
            print(text[-10000:], flush=True)
            raise RuntimeError('GATE FAIL: ' + name)
        return text
    for t in ('hip_t1_diet_parity', 'hip_qsa_fuse_parity', 'hip_node_fuse_parity', 'hip_mtp_fuse_parity'):
        test(t, [B / t])
    test('gr-parity', [B / 'gr_parity', '--selftest'])
    test('verify-parity', [B / 'verify_parity', '--selftest'])
    for mask in ('0', '12'):
        test('route-' + mask, [B / 'hip_route_plan_parity'], {'STRATA_T1_DIET': mask})
    for name, extra, args in (('hc-random', {}, []), ('hc-layer0', {}, ['--pack', PACK, '--layer', 0]),
                              ('hc-layer40-card1', {'HIP_VISIBLE_DEVICES': '1'}, ['--pack', PACK, '--layer', 40])):
        text = test(name, [B / 'hc_i8_bench', '300', '32', '1,2,3,4,5,6,7,8', *args], extra)
        for line in text.splitlines():
            if line.startswith(('PARITY', 'ERROR SUMMARY', 'BEST')):
                print(name, line, flush=True)
        if 'hc_i8_bench: PARITY OK' not in text or 'fp32/32' not in text:
            raise RuntimeError('GATE FAIL: missing HC v2/fp32-scale parity verdict')
    source_gate()
    if git('rev-parse', 'HEAD') != head or inputs() != before:
        raise RuntimeError('GATE FAIL: source/main/config/vendor changed during build/tests')
    binaries = {str(B / t): sha(B / t) for t in TARGETS}
    receipt = {'commit': head, 'inputs': before, 'binaries': binaries, 'cache': sha(B / 'CMakeCache.txt'),
               'tests': 'PASS', 'logs': str(out), 'env': L_ENV}
    (H / 'staging/live-build.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (H / 'staging/live-build.commit').write_text(head + '\n')
    (H / 'staging/live-build.sha256').write_text(''.join(f'{h}  {p}\n' for p, h in binaries.items()))
    print('LIVE_BUILD_AND_PARITY PASS', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('build', 'host-checks'))
    a = p.parse_args()
    S.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(S), CCACHE_TEMPDIR=str(S))
    os.environ['PATH'] = ':'.join(s for s in os.environ['PATH'].split(':') if s != '/home/ronald_rump/.venv/bin')
    (build if a.action == 'build' else host_checks)()
