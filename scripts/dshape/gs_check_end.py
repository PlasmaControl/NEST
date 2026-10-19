"""Independent check of a force-only run's end state, and DESC's own error.

For a run of optimize_qa.py --bext desc (tokamak, L = M = 24):

1. F on a finer collocation grid (--check x the resolution of F) and a finer box
   (h = a / --check-frac), with B_ext = B_DESC - B_plasma(x0) recomputed on that
   box (B(x0) = B_DESC at the new points) for a --bext desc run, else the coils'
   field there: <|F|> / norm overall and by rho at the
   start x0 and the end x, and DESC's own (J_DESC x B_DESC - grad p_DESC).
2. DESC's error: DESC at higher resolution (--ref, e.g. L = M = 32) at the same
   physical points (mapped to its flux coordinates by Newton), and the volume rms
   differences, over the rms of the reference, of B, J (also split along and
   across B_ref) and p for the pairs L24 - ref, NEST end - ref and NEST end - L24.
   If NEST's end is no farther from the reference than DESC L24 is, NEST's lower
   residual is not bought by moving away from the equilibrium.

usage: python scripts/dshape/gs_check_end.py --run results/gs/fdrift64a20gpu_L24_f2
    [--check 3] [--check-frac 30] [--ref desc_input/tokamak_free_boundary/tokamak_L32M32.h5]
"""

import argparse
import gc
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
from cartesian_grid import cylindrical_to_cartesian  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from boxes import make_box  # noqa: E402
from force import ForceResidual  # noqa: E402
from grid import CartesianGrid, map_to_flux  # noqa: E402
from objective import Fields  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)
REF = os.path.join(ROOT, "desc_input", "tokamak_free_boundary", "tokamak_L32M32.h5")


def gpu_peak_tracker(every=2.0):
    """Start polling nvidia-smi (memory used on this job's GPU); returns a function
    giving the peak so far in GB (nan without a GPU)."""
    import subprocess
    import threading
    peak = [float("nan")]

    def poll():
        while True:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                                      "--format=csv,noheader,nounits"], capture_output=True,
                                     text=True, timeout=10).stdout.split(",")
                used, total = float(out[0]) / 1024, float(out[1]) / 1024
                peak[0] = used if not peak[0] >= used else peak[0]
                peak.append(total) if len(peak) == 1 else None
            except Exception:
                return
            time.sleep(every)

    threading.Thread(target=poll, daemon=True).start()
    return lambda: (peak[0], peak[1] if len(peak) > 1 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--check", type=float, default=3, help="collocation / resolution of F")
    ap.add_argument("--check-frac", type=int, default=30)
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--check-grid", default="cartesian", choices=["cartesian", "cylindrical"],
                    help="grid of the check's B_plasma (the start and end are also evaluated with "
                         "the accurate axisymmetric field with --accurate)")
    ap.add_argument("--accurate", action="store_true",
                    help="B_plasma at the check points by the accurate axisymmetric field (no grid)")
    ap.add_argument("--slab", type=int, default=16, help="radii of the check grid at a time")
    ap.add_argument("--grid-res", type=int, nargs=3, default=None,
                    help="the check grid's (L, M, N) for every run alike (a common grid for "
                         "comparing runs; default: --check x the run's resolution of F)")
    ap.add_argument("--quad", type=int, nargs=2, default=(256, 96),
                    help="with --accurate: the exact field's quadrature (n_gamma, n_s); doubling it "
                         "tests the quadrature the exact-G runs share with the default check")
    ap.add_argument("--tag", default="", help="suffix of the output names (check_<run><tag>.json)")
    ap.add_argument("--save", action="store_true",
                    help="also save the points and the fields of DESC L24, the reference, the "
                         "start and the end (results/gs/fields_<run>.npz), for gs_ladder.py")
    ap.add_argument("--chunk", type=int, default=16384,
                    help="points per chunk of the box interpolations (GPU memory)")
    a = ap.parse_args()
    t0 = time.time()
    peak = gpu_peak_tracker()
    run = json.load(open(a.run + "_iter.json"))
    args, hist = run["args"], run["history"]
    st = np.load(a.run + "_iter.npz")
    x0, x1 = jnp.asarray(st["x0"]), jnp.asarray(st["x"])
    inp = DESCInput.load(args["eq"], field=args["coils"])
    eq = inp.eq
    ep = args.get("edge_power", 2)
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(args["factors"]),
                        **({"edge_power": ep} if ep != 2 else {}))
    pre = fit_cache.fit(Pressure, inp, factors=tuple(args["factors"]), edge_order=2)
    b = cur.basis
    res = lambda f: tuple(int(round(f * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))  # noqa: E731
    box = make_box(inp, a.check_frac, a.check_grid, bool(eq.sym), args.get("margin", 2.0))
    # the check grid in radial slices of --slab radii (GPU memory); the box side (flux
    # grid, box interpolation, kernel) is built once and shared
    grid_res = tuple(a.grid_res) if a.grid_res else res(a.check)
    n_r = grid_res[0] // 2 + 1
    keys = ("x", "phi", "dV", "rtz")
    G = {k: [] for k in keys}
    desc, f0, f1 = {}, {}, {}
    base = norm = None
    for i in range(0, n_r, a.slab):
        fr = ForceResidual(inp, box, resolution=grid_res, rho_slice=slice(i, i + a.slab))
        trick = args.get("bext") == "desc"
        if trick:  # replaced below: the coil field is not needed
            fr.B_ext = np.zeros((len(fr.rtz), 3))
        fields = Fields(fr, cur, pre, kernel=args["kernel"], dtype=jnp.float64, chunk=a.chunk,
                        share=base)
        base = fields if base is None else base
        d = {k: np.asarray(v) for k, v in fr.desc_fields().items()}
        if trick:  # else the coils' field (the run's own B_ext) at these points
            fields.B_ext = jnp.asarray(d["B"]) - fields.compute(x0)["B_plasma"]
        for out_, x_ in ((f0, x0), (f1, x1)):
            for k, v in fields.compute(x_).items():
                out_.setdefault(k, []).append(np.asarray(v))
        for k, v in d.items():
            desc.setdefault(k, []).append(v)
        for k in keys:
            G[k].append(np.asarray(getattr(fr, k)))
        norm, grid_shape = fr.norm, fr.tgrid.shape[1:]
        if fields is not base:
            del fields
        del fr
        gc.collect()
    desc, f0, f1 = ({k: np.concatenate(v) for k, v in dd.items()} for dd in (desc, f0, f1))
    G = {k: np.concatenate(v) for k, v in G.items()}
    if args.get("bplasma") == "axisym" or a.accurate:  # B from the coils and the accurate plasma field, no box
        from axisym_field import plasma_field_cached
        n_u = cur.n_free
        for ff, xx in ((f0, x0), (f1, x1)):
            t1 = time.time()
            B_pl = np.asarray(cylindrical_to_cartesian(
                plasma_field_cached(cur, G["rtz"], u=np.asarray(xx[:n_u]), n_gamma=a.quad[0], n_s=a.quad[1]),
                G["phi"]))
            B_coil = ff["B"] - ff["B_plasma"]
            box_err = np.sqrt(np.sum(G["dV"] * np.sum((ff["B_plasma"] - B_pl) ** 2, 1)) / np.sum(G["dV"]))
            ff["B_plasma"], ff["B"] = B_pl, B_coil + B_pl
            ff["F"] = np.cross(ff["J"], ff["B"]) - ff["grad_p"]
            print(f"accurate axisymmetric B_plasma on the check grid: {time.time() - t1:.0f} s; the "
                  f"box's (a/{a.check_frac}) differs by {box_err:.2e} T rms", flush=True)
    pk, tot = peak()
    print(f"device {jax.devices()[0].device_kind}: peak memory used (nvidia-smi) {pk:.1f} GB "
          f"of {tot:.1f} GB", flush=True)
    print(f"{inp}\nrun {a.run}: potential L, M = {b.L}, {b.M}, run box a/{args['frac']}, "
          f"collocation {res(args['colloc'])}; check: collocation {(n_r,) + grid_shape} "
          f"({len(G['x']):,} points, slices of {a.slab} radii), box a/{a.check_frac} {box.shape}; "
          f"setup {time.time() - t0:.0f} s", flush=True)
    edges = (0.0, 0.5, 0.8, 0.9, 0.95, 1.0)

    def metrics(F):
        f = np.linalg.norm(F, axis=1) / norm
        r = G["rtz"][:, 0]
        avg = lambda m: float(np.sum(G["dV"] * f * m) / np.sum(G["dV"] * m))  # noqa: E731
        return {"mean": avg(np.ones_like(f)),
                "bins": {f"{x:g}-{y:g}": avg((r > x) & (r <= y)) for x, y in zip(edges[:-1], edges[1:])}}

    F_D = metrics(np.cross(desc["J"], desc["B"]) - desc["grad(p)"])
    rows = {"DESC (J_D x B_D - grad p_D)": F_D, "NEST start x0": metrics(f0["F"]),
            "NEST end x": metrics(f1["F"])}
    bins = list(F_D["bins"])
    print(f"\n1. <|F|> / norm on the check grid and box:\n{'':34s} {'all':>9s} "
          + " ".join(f"{k:>9s}" for k in bins))
    for k, m in rows.items():
        print(f"{k:34s} {m['mean']:9.2e} " + " ".join(f"{v:9.2e}" for v in m["bins"].values()))
    print(f"{'NEST end, on the run grid and box':34s} {hist[-1]['<|F|>/norm']:9.2e}")

    # 2. DESC at higher resolution at the same physical points
    ref = DESCInput.load(a.ref)
    from desc.grid import LinearGrid
    for name_, src in (("L24", inp), ("reference", ref)):  # the iota fit used at points
        rr = np.linspace(0.01, 1, 37)
        g = LinearGrid(rho=rr, M=src.eq.M_grid, N=src.eq.N_grid, NFP=src.NFP)
        it = g.compress(src.compute(["iota"], g)["iota"])
        print(f"   iota fit of {name_} (used for point evaluation): max error "
              f"{np.max(np.abs(src.iota(rr)[0] - it)):.1e}")
    t1 = time.time()
    rtz = np.asarray(map_to_flux(ref, G["x"]))
    ok = np.all(np.isfinite(rtz), axis=1)
    d = ref.compute_points(["J", "B", "p", "grad(p)"], rtz[ok])
    R = {k: np.full((len(rtz), 3), np.nan) for k in ("J", "B", "grad(p)")}
    for k in R:
        R[k][ok] = np.asarray(cylindrical_to_cartesian(d[k], G["phi"][ok]))
    R["p"] = np.full(len(rtz), np.nan)
    R["p"][ok] = np.asarray(d["p"]).ravel()
    print(f"\n2. reference {os.path.basename(a.ref)} at the same points: {ok.sum():,} of "
          f"{len(ok):,} mapped, {time.time() - t1:.0f} s")
    w = G["dV"] * ok
    w = w / w.sum()
    rho = G["rtz"][:, 0]
    F_ref = np.where(ok[:, None], np.cross(np.nan_to_num(R["J"]), np.nan_to_num(R["B"]))
                     - np.nan_to_num(R["grad(p)"]), 0.0)
    print(f"   reference's own <|F|> / norm at these points: "
          f"{np.sum(w * np.linalg.norm(F_ref, axis=1)) / norm:.2e}")
    rmsv = lambda v: np.sqrt(np.sum(w * np.sum(np.nan_to_num(v) ** 2, axis=-1)))  # noqa: E731
    rmss = lambda v: np.sqrt(np.sum(w * np.nan_to_num(v) ** 2))  # noqa: E731
    bh = np.nan_to_num(R["B"]) / np.maximum(np.linalg.norm(np.nan_to_num(R["B"]), axis=1), 1e-300)[:, None]
    states = {"DESC L24": {"B": desc["B"], "J": desc["J"], "p": desc["p"]},
              "NEST end": {"B": f1["B"], "J": f1["J"], "p": f1["p"]},
              "NEST start": {"B": f0["B"], "J": f0["J"], "p": f0["p"]}}
    pairs = (("DESC L24", "ref"), ("NEST end", "ref"), ("NEST start", "ref"), ("NEST end", "DESC L24"))
    print(f"   volume rms of the difference / rms of the reference (B, J, p); J split along and "
          f"across B_ref:\n   {'':24s} {'B':>9s} {'J':>9s} {'J along':>9s} {'J across':>9s} {'p':>9s}")
    out = {}
    for s1, s2 in pairs:
        A_ = states[s1]
        B_ = R if s2 == "ref" else states[s2]
        dJ = A_["J"] - B_["J"]
        par = np.sum(np.nan_to_num(dJ) * bh, 1)
        perp = np.nan_to_num(dJ) - par[:, None] * bh
        v = {"B": rmsv(A_["B"] - B_["B"]) / rmsv(R["B"]), "J": rmsv(dJ) / rmsv(R["J"]),
             "J along": rmss(par) / rmsv(R["J"]), "J across": rmsv(perp) / rmsv(R["J"]),
             "p": rmss(A_["p"] - B_["p"]) / rmss(R["p"])}
        out[f"{s1} - {s2}"] = v
        print(f"   {s1 + ' - ' + s2:24s} " + " ".join(f"{v[k]:9.2e}" for k in v))
    print("   by rho, |B - B_ref| rms / rms B_ref:")
    for s1 in ("DESC L24", "NEST end"):
        vals = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (rho > lo) & (rho <= hi) & ok
            dv = states[s1]["B"][m] - R["B"][m]
            vals.append(np.sqrt(np.sum(G["dV"][m] * np.sum(dv**2, 1)) / np.sum(G["dV"][m])) / rmsv(R["B"]))
        print(f"   {s1:24s} " + " ".join(f"{v:9.2e}" for v in vals))
    if a.save:
        fname = os.path.join(ROOT, "results", "gs", f"fields_{os.path.basename(a.run)}{a.tag}.npz")
        np.savez(fname, x=G["x"], dV=G["dV"], rtz=G["rtz"], ok=ok, norm=norm,
                 grid_res=np.array(grid_res), check_frac=a.check_frac,
                 **{f"{tag}_{q}": dd[q] for tag, dd in (("desc", desc), ("start", f0), ("end", f1))
                    for q in ("J", "B", "p")},
                 **{f"{tag}_F": dd["F"] for tag, dd in (("start", f0), ("end", f1))},
                 desc_F=np.cross(desc["J"], desc["B"]) - desc["grad(p)"],
                 **{f"ref_{q}": R[q] for q in ("J", "B", "p")})
        print(f"saved {fname}")
    name = os.path.join(ROOT, "results", "gs", f"check_{os.path.basename(a.run)}{a.tag}.json")
    with open(name, "w") as fo:
        json.dump({"args": vars(a), "force": rows, "diff": out}, fo, indent=1)
    print(f"saved {name}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
