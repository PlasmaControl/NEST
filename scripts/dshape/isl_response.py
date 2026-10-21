"""What the J weight changes in the response: for scans of different weights (isl_scan.py),
at each amplitude, from the harmonic model's fields on the collocation grid:

  - the n = 1 current J_1 = J(x) - J(x with n != 0 dropped): its rms over each surface,
    against rho (where the response current sits and how peaked it is);
  - the change of the axisymmetric current from the J target, |J_0 - J_start|, against rho;
  - the rotational transform of the axisymmetric part's field, iota(rho) = <B . grad theta*> /
    <B . grad zeta> on DESC's surfaces (theta* = theta + lambda), and where iota = -1/2 (q = 2) is;
  - <|F|> / norm against rho.

Writes results/isl/<out>.json and figures/dshape/islands/<out>.png.

run (GPU): sbatch run.sbatch scripts/dshape/isl_response.py --names a120w2 a120n1 a120w4 --out isl_response
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(ROOT, "figures", "dshape", "islands")
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from harmonic_dense import HarmonicModel, harmonics  # noqa: E402
from isl_scan import model_kwargs, setup  # noqa: E402
from perturbation import Perturbation  # noqa: E402

jax.config.update("jax_enable_x64", True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--eps", type=float, nargs="*", default=[10, 100, 1000])
    ap.add_argument("--out", default="isl_response")
    a = ap.parse_args()
    out, model, geo_cache = {}, None, None
    for name in a.names:
        summ = json.load(open(os.path.join(ROOT, "results", "isl", name + ".json")))
        sa = summ["args"]

        class A:
            pass
        for k, v in sa.items():
            setattr(A, k, v)
        inp, cur, pre, x_start, fr, run = setup(A)
        if model is None:  # the same tables for every weight (only the J weight differs)
            pert = Perturbation(inp.field, sa["m"], sa["n"])
            model = HarmonicModel(inp, cur, pre, fr, run["frac"], run.get("margin", 2.0), pert=pert,
                                  chunk=sa.get("chunk", 2), batch_u=tuple(sa.get("batch", (4, 2))),
                                  **model_kwargs(A, run, cur, fr))
            model.set_source(x_start)
            nu, npp = harmonics(cur, pre)
            nh = np.concatenate([nu, npp])
            g = fr.tgrid
            rho, th, ze = np.asarray(g.rho), np.asarray(g.theta), np.asarray(g.zeta)
            shp = g.shape
            one, zero = jnp.ones(shp), jnp.zeros(shp)
            geo = fr.geo
            grad = {k: np.asarray(geo.gradient(*v)) for k, v in
                    (("r", (one, zero, zero)), ("t", (zero, one, zero)), ("z", (zero, zero, one)))}
            lam = inp.compute_tensor(["lambda_r", "lambda_t", "lambda_z"], rho, th, ze)
            lam = {k: np.asarray(v).reshape(shp) for k, v in lam.items()}
            w3 = np.asarray(fr.dV).reshape(shp)
            J_t = model._j_target if model.j_full else None
        res = {"w_J": model_kwargs(A, run, cur, fr)["weights"]["J"]}
        for eps in a.eps:
            path = os.path.join(ROOT, "results", "isl", f"{name}_eps{eps:g}.npz")
            if not os.path.exists(path):
                continue
            x = np.asarray(np.load(path)["x"])
            x0 = np.where(nh == 0, x, 0.0)  # the axisymmetric part
            f = model.fields(x, eps)
            f0 = model.fields(x0, 0.0)
            J1 = f["J"] - f0["J"]
            surf = lambda q: np.sqrt(np.sum(w3 * q, (1, 2)) / np.sum(w3, (1, 2)))  # noqa: E731
            J1_rms = surf(np.sum(J1**2, -1))
            dJ0 = surf(np.sum((f0["J"] - (J_t[:, :, None, :] if J_t is not None else 0.0)) ** 2, -1))
            B = f0["B"]
            Bt = np.sum(B * grad["t"], -1) * (1 + lam["lambda_t"]) + np.sum(B * grad["z"], -1) * lam["lambda_z"] \
                + np.sum(B * grad["r"], -1) * lam["lambda_r"]
            Bz = np.sum(B * grad["z"], -1)
            iota = np.sum(w3 * Bt, (1, 2)) / np.sum(w3 * Bz, (1, 2))
            F = np.sum(w3 * np.linalg.norm(f["F"], axis=-1), (1, 2)) / np.sum(w3, (1, 2)) / model.norm
            o = np.argsort(iota)
            rho_q2 = float(np.interp(-0.5, iota[o], rho[o]))
            res[str(eps)] = {"J1_rms": J1_rms.tolist(), "dJ0_rms": dJ0.tolist(), "iota": iota.tolist(),
                             "rho_q2": rho_q2, "F_rho": F.tolist()}
            k = np.argmax(J1_rms)
            print(f"{name} (w_J {res['w_J']:g}) eps {eps:g}: J1 peak {J1_rms[k]:.3e} at rho {rho[k]:.3f}, "
                  f"FWHM {np.ptp(rho[J1_rms > J1_rms[k] / 2]):.3f}; rms |J0 - J_start| max {dJ0.max():.3e}; "
                  f"q = 2 at rho {rho_q2:.4f}", flush=True)
        out[name] = res
    res_path = os.path.join(ROOT, "results", "isl", a.out + ".json")
    with open(res_path, "w") as fo:
        json.dump({"rho": rho.tolist(), **out}, fo, indent=1)
    fig, ax = plt.subplots(1, 4, figsize=(24, 5.5))
    for i, name in enumerate(a.names):
        c = f"C{i}"
        for eps, ls in zip(a.eps, ("-", "--", ":")):
            r = out[name].get(str(eps))
            if r is None:
                continue
            lab = f"w_J {out[name]['w_J']:g}, eps {eps:g}"
            ax[0].semilogy(rho, np.asarray(r["J1_rms"]) / eps, ls, color=c, label=lab)
            ax[1].semilogy(rho, np.maximum(r["dJ0_rms"], 1e-30), ls, color=c, label=lab)
            ax[2].plot(rho, r["iota"], ls, color=c, label=lab)
            ax[3].semilogy(rho, r["F_rho"], ls, color=c, label=lab)
    ax[2].axhline(-0.5, color="gray", lw=0.5)
    ax[0].set_title("n = 1 current per ampere: rms over each surface")
    ax[1].set_title(r"axisymmetric current: rms $|J_0 - J_{start}|$ over each surface")
    ax[2].set_title(r"$\iota$ of the axisymmetric field (q = 2 at $\iota = -1/2$)")
    ax[3].set_title(r"$\langle|F|\rangle$ / norm over each surface")
    for x_ in ax:
        x_.set_xlabel(r"$\rho$")
        x_.axvline(0.58, color="gray", lw=0.5)
        x_.legend(fontsize=7)
    fig.suptitle("What the J weight changes in NEST's response to the 2/1 error field (a/120, x2, N = 1)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, a.out + ".png"), dpi=130)
    print(f"saved figures/dshape/islands/{a.out}.png, {res_path}")


if __name__ == "__main__":
    main()
