"""QR block-size and thread-concurrency benchmark (see bench_qr.py): jnp.linalg.qr(mode="r") of (M, n)
float64 on one device for several M, then G devices at once from G Python threads."""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
n = int(sys.argv[1])
devs = jax.devices()
qr_r = jax.jit(lambda a: jnp.linalg.qr(a, mode="r"))
for M in (32000, 64000, 128000, 256000):
    try:
        A = jax.random.normal(jax.random.PRNGKey(0), (M, n), dtype=jnp.float64)
        jax.block_until_ready(qr_r(A))
        t0 = time.time()
        jax.block_until_ready(qr_r(A))
        s = time.time() - t0
        print(f"M={M:7d} one device {s:6.2f} s  {2 * M * n**2 / s / 1e12:5.1f} TFLOP/s ({s / M * 1e6:.1f} us/krow)", flush=True)
        del A
    except Exception as e:  # noqa: BLE001
        print(f"M={M} failed: {str(e)[:100]}", flush=True)
        break
M = 32000
As = [jax.device_put(jax.random.normal(jax.random.PRNGKey(i), (M, n), dtype=jnp.float64), d) for i, d in enumerate(devs)]
jax.block_until_ready(As)
for G in (1, 2, 4, 8):
    if G > len(devs):
        break
    jax.block_until_ready([qr_r(a) for a in As[:G]])
    t0 = time.time()
    jax.block_until_ready([qr_r(a) for a in As[:G]])
    s_seq = time.time() - t0

    def work(a):
        return jax.block_until_ready(qr_r(a))

    with ThreadPoolExecutor(G) as ex:
        list(ex.map(work, As[:G]))
        t0 = time.time()
        list(ex.map(work, As[:G]))
        s_thr = time.time() - t0
    print(f"G={G}: one thread {s_seq:5.2f} s, {G} threads {s_thr:5.2f} s (one block alone: see M=32000)", flush=True)
