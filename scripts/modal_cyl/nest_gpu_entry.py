"""Modal entry: select the GPU before anything imports desc, then run a tools/ script.

nvgpu (used by desc.set_device) parses the nvidia-smi banner for "CUDA Version:", which newer drivers (610+) print as
"CUDA UMD Version:", so nvgpu.gpu_info() raises IndexError there. If that happens, supply the same fields (index, type,
mem_total, mem_used in MiB) from `nvidia-smi --query-gpu`, which is stable across driver versions.
"""
import os
import runpy
import subprocess
import sys

if os.environ.get("NEST_DEVICE", "gpu") != "gpu":  # a local CPU dry run of the job scripts
    from desc import set_device

    set_device("cpu")
    script, sys.argv = sys.argv[1], sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
    sys.exit(0)

import nvgpu  # noqa: E402


def _gpu_info_query():
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True).stdout
    out = []
    for line in q.strip().splitlines():
        i, name, tot, used = [s.strip() for s in line.split(",")]
        out.append({"index": i, "type": name, "mem_total": int(tot), "mem_used": int(used)})
    return out


try:
    nvgpu.gpu_info()
except (IndexError, ValueError) as e:
    print(f"gpu_entry: nvgpu.gpu_info() failed ({type(e).__name__}); using nvidia-smi --query-gpu", flush=True)
    nvgpu.gpu_info = _gpu_info_query

from desc import set_device  # noqa: E402

set_device("gpu")
script, sys.argv = sys.argv[1], sys.argv[1:]
runpy.run_path(script, run_name="__main__")
