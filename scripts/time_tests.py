"""Run every test file (``tests/test_*.py``, each as its own process with its
default arguments) and report pass/fail and wall time, slowest first.

run (GPU): sbatch --time=03:00:00 run.sbatch scripts/time_tests.py [name ...] [--twice]
(--twice: run everything twice, e.g. to see the gain of the compilation cache)
"""

import glob
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# tests with large boxes (a/30) need the platform allocator (no pool fragmentation)
ENV = {"test_reconstruction.py": {"XLA_PYTHON_CLIENT_ALLOCATOR": "platform"}}


def main():
    names = [a for a in sys.argv[1:] if a != "--twice"]
    for k in range(2 if "--twice" in sys.argv else 1):
        if k:
            print("\n=== second run")
        run(names)


def run(names):
    files = sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py")))
    if names:
        files = [f for f in files if any(n in os.path.basename(f) for n in names)]
    rows = []
    for f in files:
        t0 = time.time()
        env = dict(os.environ, **ENV.get(os.path.basename(f), {}))
        p = subprocess.run([sys.executable, f], cwd=ROOT, capture_output=True, text=True, env=env)
        dt = time.time() - t0
        ok = p.returncode == 0
        rows.append((os.path.basename(f), ok, dt))
        print(f"{os.path.basename(f):32s} {'pass' if ok else 'FAIL'} {dt:8.1f} s", flush=True)
        if not ok:
            print("    " + "\n    ".join((p.stdout + p.stderr).strip().splitlines()[-8:]), flush=True)
    print("\nslowest first:")
    for name, ok, dt in sorted(rows, key=lambda r: -r[2]):
        print(f"  {name:32s} {'pass' if ok else 'FAIL'} {dt:8.1f} s")
    print(f"total {sum(r[2] for r in rows):.0f} s, {sum(not r[1] for r in rows)} failed")


if __name__ == "__main__":
    main()
