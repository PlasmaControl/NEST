"""p of mf_fixed states on the planes phi = 0 and phi = pi / NFP (DESC coordinates -> R, Z).

Saves, per state: p on (rho, theta) of each plane, the surface average <p>(rho) over (theta,
zeta), and dp = p - <p>(rho) (the part of p that is not a function of DESC's rho).
usage: python pressure_sections.py --LMN 32 32 24 --state name=path.npz ... --out p.npz
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))
from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
import fit_cache  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from pressure import Pressure  # noqa: E402
from separable import Separable  # noqa: E402

D = os.path.join(ROOT, "desc_input", "vmec_free_boundary", "NCSX")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=os.path.join(D, "NCSX_fb_L16M16N12.h5"))
    ap.add_argument("--coils", default=os.path.join(D, "NCSX_coils.h5"))
    ap.add_argument("--LMN", type=int, nargs=3, required=True)
    ap.add_argument("--state", action="append", default=[])
    ap.add_argument("--n", type=int, nargs=2, default=(161, 361), help="rho, theta per plane")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    inp = DESCInput.load(a.eq, field=a.coils)
    eq = inp.eq
    pre = fit_cache.fit(Pressure, inp, edge_order=2, L=a.LMN[0], M=a.LMN[1], N=a.LMN[2])
    RZ = Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP)
    rho = np.linspace(0, 1, a.n[0]); rho[0] = 1e-4
    th = np.linspace(0, 2 * np.pi, a.n[1])
    out = {"rho": rho, "theta": th, "planes": np.array([0.0, np.pi / eq.NFP])}
    pl = []
    for k, ph in enumerate(out["planes"]):
        Rg, Tg = np.meshgrid(rho, th, indexing="ij")
        nodes = np.column_stack([Rg.ravel(), Tg.ravel(), np.full(Rg.size, ph)])
        X = np.asarray(RZ.evaluate(jnp.asarray(nodes[:, 0]), jnp.asarray(nodes[:, 1]), jnp.asarray(nodes[:, 2])))
        out[f"R{k}"], out[f"Z{k}"] = X[:, 0].reshape(Rg.shape), X[:, 1].reshape(Rg.shape)
        pl.append(nodes)
    # surface average on a full (theta, zeta) grid per rho
    tt = 2 * np.pi * np.arange(64) / 64
    zz = 2 * np.pi / eq.NFP * np.arange(32) / 32
    A, B, C = np.meshgrid(rho, tt, zz, indexing="ij")
    full = np.column_stack([A.ravel(), B.ravel(), C.ravel()])
    states = [("start", pre.c)] + [(s.split("=")[0], np.load(s.split("=")[1])["c_p"]) for s in a.state]
    for name, c in states:
        P = pre.with_c(jnp.asarray(c))
        avg = np.asarray(P.evaluate(SimpleNamespace(nodes=full)))[:, 0].reshape(A.shape).mean(axis=(1, 2))
        out[f"{name}_avg"] = avg
        for k, nodes in enumerate(pl):
            p = np.asarray(P.evaluate(SimpleNamespace(nodes=nodes)))[:, 0].reshape(len(rho), len(th))
            out[f"{name}_p{k}"], out[f"{name}_dp{k}"] = p, p - avg[:, None]
        print(f"{name}: p(axis) {avg[0]:.4e}, max |p - <p>| / p(axis): plane 0 "
              f"{np.abs(out[f'{name}_dp0']).max() / avg[0]:.3e}, plane 1 {np.abs(out[f'{name}_dp1']).max() / avg[0]:.3e}; "
              f"min p {min(out[f'{name}_p0'].min(), out[f'{name}_p1'].min()):.3e}", flush=True)
    np.savez(a.out, **out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
