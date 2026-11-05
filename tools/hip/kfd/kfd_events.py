#!/usr/bin/env python3
"""KFD SMI event logger: every queue eviction / restore / unmap / VM fault / reset KFD reports, one line each.

usage: kfd_events.py OUT [--all] [--secs N]
  --all  every process's events (needs root: KFD_SMI_EVENT_ALL_PROCESS); without it, only this process's (none).
Lines: '<CLOCK_BOOTTIME ns read here> <gpu node> <event name> <KFD payload>'; the payload's own ns is CLOCK_BOOTTIME too.
"""
import fcntl, os, select, struct, sys, time

IOC = 0xC0084B1F  # _IOWR('K', 0x1F, struct kfd_ioctl_smi_events_args {u32 gpuid; u32 anon_fd;})
NAMES = {1: 'VMFAULT', 2: 'THERMAL', 3: 'PRE_RESET', 4: 'POST_RESET', 5: 'MIGRATE_START', 6: 'MIGRATE_END',
         7: 'PF_START', 8: 'PF_END', 9: 'QUEUE_EVICTION', 10: 'QUEUE_RESTORE', 11: 'UNMAP_FROM_GPU',
         12: 'PROC_START', 13: 'PROC_END'}
EVICT_TRIG = {0: 'SVM', 1: 'USERPTR', 2: 'TTM', 3: 'SUSPEND', 4: 'CRIU_CKPT', 5: 'CRIU_RESTORE'}

def bit(i): return 1 << (i - 1)

def gpus():
    base = '/sys/class/kfd/kfd/topology/nodes'
    out = []
    for n in sorted(os.listdir(base), key=int):
        gid = int(open(f'{base}/{n}/gpu_id').read().strip() or 0)
        if gid:
            loc = 0
            for l in open(f'{base}/{n}/properties'):
                if l.startswith('location_id'): loc = int(l.split()[1])
            out.append((int(n), gid, loc))
    return out

def main():
    out = sys.argv[1]; allp = '--all' in sys.argv
    secs = float(sys.argv[sys.argv.index('--secs') + 1]) if '--secs' in sys.argv else None
    kfd = os.open('/dev/kfd', os.O_RDWR | os.O_CLOEXEC)
    mask = (bit(1) | bit(3) | bit(4) | bit(9) | bit(10) | bit(11)) | ((1 << 63) if allp else 0)
    fds = {}
    for node, gid, loc in gpus():
        buf = bytearray(struct.pack('II', gid, 0))
        fcntl.ioctl(kfd, IOC, buf, True)
        fd = struct.unpack('II', bytes(buf))[1]
        os.write(fd, struct.pack('Q', mask))
        fds[fd] = (node, f'{loc >> 8:02x}:{(loc >> 3) & 31:02x}.{loc & 7}')
    t_end = time.time() + secs if secs else None
    with open(out, 'a', buffering=1) as f:
        f.write(f'# kfd_events start {time.strftime("%H:%M:%S")} boottime_ns {time.clock_gettime_ns(time.CLOCK_BOOTTIME)} '
                f'all={allp} gpus {sorted(fds.values())}\n')
        p = select.poll()
        for fd in fds: p.register(fd, select.POLLIN)
        while t_end is None or time.time() < t_end:
            for fd, _ in p.poll(1000):
                data = os.read(fd, 4096).decode(errors='replace')
                now = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
                for line in data.splitlines():
                    if not line.strip(): continue
                    head, _, rest = line.partition(' ')   # "%x %s": the event id in hex, then its payload
                    try: eid = int(head, 16)
                    except ValueError: eid, rest = 0, line
                    extra = ''
                    if eid == 9:
                        try: extra = ' trigger=' + EVICT_TRIG.get(int(rest.split()[-1]), rest.split()[-1])
                        except Exception: pass
                    f.write(f'{now} node{fds[fd][0]}({fds[fd][1]}) {NAMES.get(eid, eid)} {rest}{extra}\n')

if __name__ == '__main__':
    main()
