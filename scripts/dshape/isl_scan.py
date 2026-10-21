"""Island growth on the tokamak: NEST's 3-D equilibrium in the coils' field plus a
resonant error field eps B_pert (``perturbation.py``: Phi = sin(m theta - n zeta) on
the winding surface), for a sequence of amplitudes eps, each continued from the last.

The start is a converged axisymmetric run (``--base``, e.g. the cylindrical-annulus
x3 run at a/320) embedded in 3-D bases with toroidal modes |n| <= N (``--N``); the
problem is solved by the exact dense LM on harmonic tables (``harmonic_dense.py``) on
the base run's annulus spacing, collocation and weights (the J target on the toroidal
average only). ``--verify``: the model's F against the 3-D annulus solve (N_phi = 2N + 1)
at a perturbed x, before the scan.

Writes results/isl/<name>_eps<eps>.npz/.json (x, history) and <name>.json (summary).

run (GPU): sbatch --qos=gpu-stellar --time=12:00:00 --mem=300G run.sbatch scripts/isl_scan.py \
    --base results/gs/cyl3D_a320 --N 1 --eps 0 30 100 300 1000 --name n1
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from boxes import make_box  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from flux_interp import TensorCurrent, TensorGrid  # noqa: E402
from force import ForceResidual  # noqa: E402
from harmonic_dense import HarmonicModel, embed, harmonics, to_cyl  # noqa: E402
from objective import Fields  # noqa: E402
from perturbation import Perturbation  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)
TOK = os.path.join(ROOT, "desc_input", "tokamak_free_boundary")


def setup(a):
    """inp, the 3-D bases, the start x, the 3-D collocation and the base run's settings."""
    if a.base:
        run = json.load(open(a.base + "_iter.json"))["args"]
        x2 = np.asarray(np.load(a.base + "_iter.npz")["x"])
    else:  # a fresh start from the fits (tests)
        run = {"eq": a.eq, "coils": a.coils, "factors": a.factors, "edge_power": a.edge_power, "frac": a.frac,
               "margin": 2.0, "colloc": 2.0, "w_J": 1e-3, "w_J_rho": 0.9, "w_W": 10.0, "w_pos": 10.0}
        x2 = None
    inp = DESCInput.load(run["eq"], field=run["coils"])
    eq = inp.eq
    ep = run.get("edge_power", 2)
    kw = {"edge_power": ep} if ep != 2 else {}
    fac = tuple(run["factors"])
    cur3 = fit_cache.fit(BoundaryPotential, inp, factors=fac, N=a.N, **kw)
    pre3 = fit_cache.fit(Pressure, inp, factors=fac, N=a.N, edge_order=2)
    if x2 is None:
        x = np.concatenate([cur3.u, pre3.c])
        nu, npp = harmonics(cur3, pre3)
        x[np.concatenate([nu, npp]) > 0] = 0.0  # axisymmetric start
    else:
        cur2 = fit_cache.fit(BoundaryPotential, inp, factors=fac, **kw)
        pre2 = fit_cache.fit(Pressure, inp, factors=fac, edge_order=2)
        x = embed(x2, cur2, pre2, cur3, pre3)
    b = cur3.basis
    res = tuple(int(round(run["colloc"] * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    fr = ForceResidual(inp, None, resolution=res)
    return inp, cur3, pre3, x, fr, run


def model_kwargs(a, run, cur, fr):
    """The base run's objective for ``HarmonicModel``: its weights, and with j_target "start"
    the full J against J at the base run's start (embedded in the 3-D bases) at every point."""
    weights = {"J": run.get("w_J", 1e-3), "J_rho": run.get("w_J_rho", 0.9), "W": run.get("w_W", 10.0),
               "pos": run.get("w_pos", 10.0)}
    if getattr(a, "w_J", None) is not None:  # the J weight changed from the base run's
        weights["J"] = a.w_J
    if getattr(a, "w_W", None) is not None:
        weights["W"] = a.w_W
    kw = {"weights": weights, "j_nodes": run.get("j_nodes") or "flux", "j_mode": getattr(a, "j_mode", "full")}
    if getattr(a, "transport", None):
        kw["transport"] = {"kappa": a.transport, "w": a.w_T}
    if getattr(a, "par", None):
        kw["par"] = a.par
    if getattr(a, "rough", None):
        kw["rough"] = {"w": a.rough, "q": a.rough_q}
    if a.base and run.get("j_target") == "start":
        inp = fr.inp
        ep = run.get("edge_power", 2)
        e = {"edge_power": ep} if ep != 2 else {}
        fac = tuple(run["factors"])
        cur2 = fit_cache.fit(BoundaryPotential, inp, factors=fac, **e)
        pre2 = fit_cache.fit(Pressure, inp, factors=fac, edge_order=2)
        x0 = embed(np.asarray(np.load(a.base + "_iter.npz")["x0"]), cur2, pre2, cur, fit_cache.fit(
            Pressure, inp, factors=fac, N=a.N, edge_order=2))
        g = TensorGrid(np.asarray(fr.tgrid.rho), np.asarray(fr.tgrid.theta), np.array([0.0]), inp.eq.NFP)
        tc = TensorCurrent(cur.S, cur._RZ, g, factored=True)
        kw["j_target"] = np.asarray(tc(jnp.asarray(x0[: cur.n_free]), cur.I))[:, :, 0]
    return kw


def verify(model, inp, cur, pre, fr, run, x, eps, pert):
    """max |F_model - F_annulus| / max |F| at x, with the 3-D annulus solve (N_phi = 2N + 1)."""
    t0 = time.time()
    b = cur.basis
    N_phi = 2 * model.N + 1
    box = make_box(inp, run["frac"], "cylindrical", True, run.get("margin", 2.0), verbose=True, N_phi=N_phi)
    fr.box = box
    fg = TensorGrid.uniform(4 * b.L, 2 * (3 * b.M), 4 * N_phi, inp.eq.NFP, sym=True)
    fields = Fields(fr, cur, pre, kernel="cylindrical", dtype=jnp.float64, flux_grid=fg, factored=True,
                    S_dtype=jnp.float64)
    if run.get("j_nodes") == "direct":  # J at the nodes from the basis, as the run
        from node_current import NodeFields
        fields = NodeFields.from_fields(fields, cur, box)
    fields.B_ext = jnp.asarray(np.asarray(fr.B_ext) + eps * np.asarray(pert(fr.x)))
    f = {k: np.asarray(v) for k, v in fields.compute(jnp.asarray(x)).items()}
    m = model.fields(x, eps)
    phi = np.asarray(fr.phi)
    out = {}
    for k_m, k_f in (("F", "F"), ("J", "J"), ("B", "B")):
        Vm = to_cyl(f[k_f], phi).reshape(m[k_m].shape)
        out[k_m] = float(np.abs(Vm - m[k_m]).max() / np.abs(Vm).max())
    w = np.asarray(fields.sw) ** 2
    out["<|F|>/norm annulus"] = float(np.sum(w * np.linalg.norm(f["F"], axis=1)) / fields.norm)
    out["<|F|>/norm model"] = model.evaluate(x, eps)[1]
    print(f"verify at eps {eps:g} (N_phi {N_phi}, {time.time() - t0:.0f} s): max error / max of F "
          f"{out['F']:.1e}, J {out['J']:.1e}, B {out['B']:.1e}; <|F|>/norm annulus "
          f"{out['<|F|>/norm annulus']:.6e}, model {out['<|F|>/norm model']:.6e}", flush=True)
    fr.box = None
    del fields, box
    jax.clear_caches()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help="converged axisymmetric run to start from (its _iter files)")
    ap.add_argument("--eq", default=os.path.join(TOK, "tokamak_L32M32_fb.h5"))
    ap.add_argument("--coils", default=os.path.join(TOK, "tokamak_L32M32_field.h5"))
    ap.add_argument("--factors", type=float, nargs=3, default=(3, 3, 1))
    ap.add_argument("--edge-power", type=int, default=3)
    ap.add_argument("--frac", type=int, default=320)
    ap.add_argument("--N", type=int, default=1, help="toroidal modes |n| <= N of J and p")
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--n", type=int, default=-1, help="Phi = sin(m theta - n zeta): n = -1 resonates on q = 2")
    ap.add_argument("--eps", type=float, nargs="+", default=[0, 30, 100, 300, 1000], help="amplitudes [A]")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--rtol", type=float, default=1e-7)
    ap.add_argument("--gn-tol", type=float, default=1e-6,
                    help="stop when a full Gauss-Newton step could remove less than this share of f")
    ap.add_argument("--steps", default="qr", choices=["qr", "lsqr"])
    ap.add_argument("--predict", type=float, default=None,
                    help="start each amplitude from the response at this solved one scaled linearly, "
                         "x0 + (eps / eps_ref) (x_ref - x0) (x0: the eps = 0 state); the amplitudes are then "
                         "independent (separate jobs can run them)")
    ap.add_argument("--chunk", type=int, default=2, help="radii per device chunk")
    ap.add_argument("--batch", type=int, nargs=2, default=(4, 2), help="box solves at a time (n = 0, n > 0)")
    ap.add_argument("--j-n0", action="store_true", default=True)
    ap.add_argument("--verify", type=float, default=None, help="check the model against the annulus at this eps")
    ap.add_argument("--only", nargs="*", default=None,
                    help="build and cache these tables (u0 u1 p0 p1 ...) and the constants, then stop")
    ap.add_argument("--again", action="store_true", help="with --resume: continue the finished amplitudes too")
    ap.add_argument("--resume", action="store_true",
                    help="skip the amplitudes already done; continue an unfinished one from its last x")
    ap.add_argument("--start-from", default=None,
                    help="another scan's name (e.g. n1, smaller N): each amplitude starts from its x there, "
                         "embedded in these bases, when this scan has none yet")
    ap.add_argument("--w-J", type=float, default=None, help="the J term's weight (default: the base run's)")
    ap.add_argument("--w-W", type=float, default=None, help="the <p> term's weight (default: the base run's)")
    ap.add_argument("--j-mode", default="full", choices=["full", "n0"],
                    help="the J term on the full J (as the 2-D runs) or on its toroidal average only")
    ap.add_argument("--transport", type=float, default=None,
                    help="kappa_par / kappa_perp: pressure from anisotropic transport in the computed field "
                         "(weak form on the pressure basis, source from the eps = 0 state)")
    ap.add_argument("--w-T", type=float, default=1.0, help="weight of the transport rows")
    ap.add_argument("--rough", type=float, default=None,
                    help="weight of the n >= 1 current smoothness rows, (l / L)^q per coefficient")
    ap.add_argument("--rough-q", type=float, default=2.0)
    ap.add_argument("--par", type=float, default=None,
                    help="weight of the force component along B (B . grad p = 0) against the rest of F")
    ap.add_argument("--name", default="isl")
    a = ap.parse_args()
    t0 = time.time()
    out_dir = os.path.join(ROOT, "results", "isl")
    os.makedirs(out_dir, exist_ok=True)
    inp, cur, pre, x, fr, run = setup(a)
    x_base = np.asarray(x)
    pert = Perturbation(inp.field, a.m, a.n)
    print(f"{inp}\n{pert}\nbase {a.base}: h = a/{run['frac']}, factors {run['factors']}, N = {a.N}; "
          f"collocation {fr.tgrid.shape}; setup {time.time() - t0:.0f} s", flush=True)
    model = HarmonicModel(inp, cur, pre, fr, run["frac"], run.get("margin", 2.0), pert=pert,
                          chunk=a.chunk, batch_u=tuple(a.batch), only=a.only, **model_kwargs(a, run, cur, fr))
    if a.only:
        print(f"tables {a.only} built; total {time.time() - t0:.0f} s")
        return
    model.set_source(x_base)  # the transport source: the eps = 0 (unperturbed) state
    f0, F0 = model.evaluate(x, 0.0)
    base_f = None
    if a.base:
        base_f = json.load(open(a.base + "_iter.json"))["history"][-1]
    print(f"start (eps 0): f {f0:.6e}, <|F|>/norm {F0:.6e}"
          + (f"; the base run's own: f {base_f['f']:.6e}, <|F|>/norm {base_f['<|F|>/norm']:.6e}" if base_f else ""),
          flush=True)
    summary = {"args": vars(a), "base_args": run, "start": {"f": f0, "<|F|>/norm": F0}, "runs": []}
    spath = os.path.join(out_dir, a.name + ".json")
    if a.resume and os.path.exists(spath):
        old = json.load(open(spath))
        summary["runs"] = old.get("runs", [])
        if "verify" in old:
            summary["verify"] = old["verify"]
    if a.verify is not None and "verify" not in summary:
        rng = np.random.default_rng(0)
        nu, npp = harmonics(cur, pre)
        xt = x + 1e-3 * np.abs(x).max() * rng.standard_normal(len(x)) * (np.concatenate([nu, npp]) > 0)
        summary["verify"] = verify(model, inp, cur, pre, fr, run, xt, a.verify, pert)
    for eps in a.eps:
        t1 = time.time()
        tag = f"{a.name}_eps{eps:g}"
        path = os.path.join(out_dir, tag)
        prev = []
        other = os.path.join(out_dir, f"{a.start_from}_eps{eps:g}") if a.start_from else None
        if other and os.path.exists(other + ".npz") and not os.path.exists(path + ".json"):
            o = json.load(open(other + ".json"))
            N0 = int(np.load(other + ".npz")["N"])
            ep = run.get("edge_power", 2)
            kw = {"edge_power": ep} if ep != 2 else {}
            c0 = fit_cache.fit(BoundaryPotential, inp, factors=tuple(run["factors"]), N=N0, **kw)
            p0 = fit_cache.fit(Pressure, inp, factors=tuple(run["factors"]), N=N0, edge_order=2)
            x = embed(np.asarray(np.load(other + ".npz")["x"]), c0, p0, cur, pre)
            print(f"eps {eps:g} A: starting from {a.start_from} (N = {N0}, <|F|>/norm "
                  f"{o['history'][-1]['<|F|>/norm']:.3e} there)", flush=True)
        if a.predict and eps not in (0.0, a.predict) and not os.path.exists(path + ".json"):
            ref = os.path.join(out_dir, f"{a.name}_eps{a.predict:g}.npz")
            zero = os.path.join(out_dir, f"{a.name}_eps0.npz")
            xr = np.asarray(np.load(ref)["x"])
            x0 = np.asarray(np.load(zero)["x"]) if os.path.exists(zero) else x_base
            x = x0 + (eps / a.predict) * (xr - x0)
            print(f"eps {eps:g} A: starting from the linear response of eps {a.predict:g}", flush=True)
        if a.resume and os.path.exists(path + ".json"):
            d = json.load(open(path + ".json"))
            x = np.asarray(np.load(path + ".npz")["x"])
            if d.get("done") and not a.again:
                print(f"eps {eps:g} A: done before, <|F|>/norm {d['history'][-1]['<|F|>/norm']:.3e}", flush=True)
                continue
            prev = d["history"]
            print(f"eps {eps:g} A: continuing after {len(prev) - 1} iterations", flush=True)

        def save(xx, hist, done=False, path=path, eps=eps, prev=prev):
            h = prev + [dict(e, it=e["it"] + max(len(prev) - 1, 0)) for e in (hist[1:] if prev else hist)]
            np.savez(path + ".npz", x=np.asarray(xx), eps=eps, N=a.N)
            with open(path + ".json", "w") as fo:
                json.dump({"args": vars(a), "base_args": run, "eps": eps, "history": h, "done": done}, fo, indent=1)
            return h

        x, hist = model.lm(x, eps, iters=a.iters, rtol=a.rtol, callback=save, steps=a.steps, gn_tol=a.gn_tol)
        hist = save(x, hist, done=True)
        summary["runs"] = [r for r in summary["runs"] if r["eps"] != eps]
        summary["runs"].append({"eps": eps, "f": hist[-1]["f"], "<|F|>/norm": hist[-1]["<|F|>/norm"],
                                "iters": len(hist) - 1, "time": time.time() - t1})
        summary["runs"].sort(key=lambda r: r["eps"])
        print(f"eps {eps:g} A: <|F|>/norm {hist[0]['<|F|>/norm']:.3e} -> {hist[-1]['<|F|>/norm']:.3e} in "
              f"{len(hist) - 1} iterations, {time.time() - t1:.0f} s", flush=True)
        with open(os.path.join(out_dir, a.name + ".json"), "w") as fo:
            json.dump(summary, fo, indent=1)
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
