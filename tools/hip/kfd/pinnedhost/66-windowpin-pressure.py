#!/usr/bin/env python3
"""Window PIN's owned pressure workers; no subprocesses or persistent settings."""
import signal, sys, time
stop = False
def finish(*_):
    global stop
    stop = True
signal.signal(signal.SIGTERM, finish)
signal.signal(signal.SIGINT, finish)
if sys.argv[1] == 'compact':
    while not stop:
        with open('/proc/sys/vm/compact_memory', 'w') as f:
            f.write('1\n')
        end = time.monotonic() + 10
        while not stop and time.monotonic() < end:
            time.sleep(.1)
else:
    while not stop:
        for path in sys.argv[2:]:
            with open(path, 'rb', buffering=0) as f:
                while not stop:
                    got = 0
                    for _ in range(16):
                        block = f.read(16 << 20)
                        got += len(block)
                        if not block or stop: break
                    if not got: break
                    time.sleep(.25)
            if stop: break
