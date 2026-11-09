"""Keeps Windows from idle-sleeping while a long local run is going, without touching any power setting.

It asks for the system-required execution state (the request media players make) and holds it until the run
log shows the run has ended, or for at most --hours. The request ends when this process exits. A closed lid or
a manual sleep still sleeps.

    python scripts/stay_awake.py runs/main.log --hours 16
"""

import argparse
import ctypes
import sys
import time
from pathlib import Path

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
DONE = ("# Results (", "giving up on:", "budget reached:")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--hours", type=float, default=16)
    a = p.parse_args()
    if sys.platform != "win32":
        print("not Windows; nothing to do")
        return 0
    k32 = ctypes.windll.kernel32
    k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    start, log = time.time(), Path(a.log)
    offset = log.stat().st_size if log.exists() else 0     # only what the resumed run writes counts
    print(f"holding the system awake until {a.log} shows the end of the run (at most {a.hours} h)", flush=True)
    try:
        while time.time() - start < a.hours * 3600:
            text = log.read_bytes()[offset:].decode("utf-8", errors="replace") if log.exists() else ""
            if any(d in text for d in DONE):
                print("run ended", flush=True)
                break
            time.sleep(60)
    finally:
        k32.SetThreadExecutionState(ES_CONTINUOUS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
