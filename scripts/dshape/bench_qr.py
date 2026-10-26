"""Micro-benchmark of the TSQR building block on the available GPUs: R of a (M, n) float64 block by
jnp.linalg.qr(mode="r") against lax.linalg.geqrf (no explicit Q), float32, and G devices at once."""
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
M, n = int(sys.argv[1]), int(sys.argv[2])
devs = jax.devices()
print("devices", devs, flush=True)


def t(fn, *a, rep=3):
    jax.block_until_ready(fn(*a))
    t0 = time.time()
    for _ in range(rep):
        out = fn(*a)
    jax.block_until_ready(out)
    return (time.time() - t0) / rep


qr_r = jax.jit(lambda a: jnp.linalg.qr(a, mode="r"))


geqrf_r = qr_r


key = jax.random.PRNGKey(0)
A = jax.random.normal(key, (M, n), dtype=jnp.float64)
flops = 2 * M * n**2 - 2 * n**3 / 3
for name, fn in (("jnp.linalg.qr r", qr_r),):
    s = t(fn, A)
    print(f"{name:16s} fp64 {s:6.2f} s  {flops / s / 1e12:5.1f} TFLOP/s", flush=True)
A32 = A.astype(jnp.float32)
for name, fn in (("jnp.linalg.qr r", qr_r),):
    s = t(fn, A32)
    print(f"{name:16s} fp32 {s:6.2f} s  {flops / s / 1e12:5.1f} TFLOP/s", flush=True)
# G devices at once (independent blocks, one Python thread, async dispatch)
for G in (1, 2, 4, 8):
    if G > len(devs):
        break
    As = [jax.device_put(A, d) for d in devs[:G]]
    jax.block_until_ready(As)
    for name, fn in (("jnp.linalg.qr r", qr_r),):
        jax.block_until_ready([fn(a) for a in As])
        t0 = time.time()
        for _ in range(3):
            outs = [fn(a) for a in As]
        jax.block_until_ready(outs)
        s = (time.time() - t0) / 3
        print(f"G={G} {name:16s} {s:6.2f} s for {G} blocks ({s / G:5.2f} s per block)", flush=True)
