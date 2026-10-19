"""Run a NEST optimization on the smooth-current precise_QA in its coils.

Starts from the DESC equilibrium (``BoundaryPotential`` and ``Pressure`` fits),
minimizes the chosen weighted objectives with the matrix-free Levenberg-Marquardt
of ``optimize``, and saves the coefficients, the history and the force error by
radius before and after to ``--out`` (npz) and the history to ``--out`` .json.

Default: the current and the pressure free, the stored energy held and p >= 0,
the truncated kernel, the block preconditioner; ``--refine 20`` continues on a
finer box.

run (GPU): sbatch --qos=gpu-stellar --time=03:00:00 run.sbatch scripts/optimize_qa.py \
    [--frac 10] [--refine 20] [--kernel hockney|truncated] [--grid cartesian|cylindrical] [--free current|all] \
    [--maxiter 100] [--cg 100] [--precond jacobi|block] [--time-limit s] [--start x.npz] \
    [--w-force 1] [--w-B 0] [--w-J 0] [--w-p 0] [--w-W 10] [--w-p2 0] [--w-pos 10] [--factors 3 3 2.5] \
    [--out results/opt]

Force-only drift study from a start that matches DESC (axisymmetric tokamak, fast on CPU):
    sbatch run_cpu.sbatch scripts/optimize_qa.py --eq T.h5 --coils T_field.h5 --bext desc \
        --fit force --dense --no-check --w-W 0 --w-pos 0 --ftol 0 --track [--factors 2 2 1]
--bext desc and --fit force make F_NEST(x0) equal DESC's residual on the points (up to
the force-matched fit, dense.py); --dense precomputes B_plasma(u) as a matrix (cached).
"""

import argparse
import functools
import gc
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from grid import CartesianGrid  # noqa: E402
from objective import (CurrentError, CurrentMoments, FieldError, Fields, ForceBalance,  # noqa: E402
                       ObjectiveFunction, PressureEnergy, PressureError,
                       PressurePositivity, PressureSquare, _cartesian)
from axisym_field import plasma_field_cached  # noqa: E402
from cartesian_grid import cylindrical_to_cartesian  # noqa: E402
from dense import DenseFields, force_matched_u, j_target  # noqa: E402
from optimize import levenberg_marquardt  # noqa: E402
from precondition import BlockPreconditioner, mode_groups, mode_labels  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)
EQ = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_smooth_J",
                  "precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5")
COILS = os.path.join(ROOT, "desc_input", "precise_QA_free_boundary", "precise_QA_helical_coils.h5")


def make_box(inp, frac, grid="cartesian", sym=True, margin=2.0):
    """The B_plasma grid at h = a / frac: main's periodic Cartesian box, or the
    cylindrical annulus (``cylindrical_grid``), with the plasma located on it."""
    if grid == "cylindrical":
        from cylindrical_grid import CylindricalGrid
        g = CylindricalGrid.around(inp, inp.a / frac, margin)
        zs = g.z_axis
        sym_z = abs(zs.a + zs.b) <= 1e-12 * max(abs(zs.a), abs(zs.b))
        return g.locate_plasma(inp, verbose=True, sym=sym and sym_z)
    return CartesianGrid.periodic_box(inp, inp.a / frac, odd=sym).locate_plasma(inp)


def check(inp, box, cur, pre, resolution, kernel, xs, n_rho=8):
    """<|F|>_vol / norm (overall and by radius) of each x in xs on the collocation
    grid of ``resolution``, evaluated n_rho radii at a time (the memory of a
    fraction of the grid); returns the metrics and the grid's shape."""
    full = ForceResidual(inp, box, resolution=resolution, rho_slice=slice(0, 1))
    L = resolution[0]
    n_r = L // 2 + 1
    F = {k: [] for k in xs}
    dV, rho = [], []
    base = None  # the box side (flux grid, box interpolation, kernel), built once
    for i in range(0, n_r, n_rho):
        fr = ForceResidual(inp, box, resolution=resolution, rho_slice=slice(i, i + n_rho))
        fields = Fields(fr, cur, pre, kernel=kernel, share=base)
        base = fields if base is None else base
        for k, x in xs.items():
            F[k].append(np.linalg.norm(np.asarray(fields.compute(x)["F"]), axis=1) / fr.norm)
        dV.append(fr.dV)
        rho.append(fr.rtz[:, 0])
        del fields, fr
        gc.collect()
    del base
    jax.clear_caches()
    gc.collect()
    dV, rho = np.concatenate(dV), np.concatenate(rho)
    edges = (0.0, 0.5, 0.8, 0.9, 0.95, 1.0)
    avg = lambda f, m: float(np.sum(f * dV * m) / np.sum(dV * m))
    out = {}
    for k in xs:
        f = np.concatenate(F[k])
        out[k] = {"mean": avg(f, 1.0), "bins": {f"{a:g}-{b:g}": avg(f, (rho > a) & (rho <= b))
                                               for a, b in zip(edges[:-1], edges[1:])}}
    return out, (n_r,) + full.tgrid.shape[1:]


def drift(fields, x, x0, ref=None):
    """Where x has gone from x0: relative change of u and c_p, <p>, sqrt <p^2>, min p,
    rms J, the net toroidal current (V <J_phi / R> / 2 pi), all relative to x0;
    rms |B - B_DESC| / rms |B_DESC|, how far p is from a flux function
    (<|B.grad p|> / <|B| |grad p|>), how far B crosses DESC's nested surfaces
    (<|B.grad rho|> / <|B| |grad rho|>, also for rho > 0.9) and <|F|> / norm.
    ``ref``: the fields at x0."""
    f = {k: np.asarray(v) for k, v in fields.compute(x).items()}
    ref = f if ref is None else ref
    w = np.asarray(fields.sw) ** 2
    avg = lambda q: float(np.sum(w * q))  # noqa: E731
    R, ph = np.asarray(fields.force.geo.R).ravel(), np.asarray(fields.phi)
    I = lambda J: avg((-J[:, 0] * np.sin(ph) + J[:, 1] * np.cos(ph)) / R)  # noqa: E731
    rms = lambda v: np.sqrt(avg(np.sum(v * v, 1)))  # noqa: E731
    x, x0, n_u = np.asarray(x), np.asarray(x0), fields.n_u
    B2 = np.sum(f["B"] ** 2, 1)
    gp = f["grad_p"]
    # grad rho of DESC's surfaces ~ e_theta x e_zeta = (-R Z_t, Z_t R_z - R_t Z_z, R R_t)
    g = fields.force.geo
    Xt, Xz = (np.asarray(v).reshape(-1, 2) for v in (g.Xt, g.Xz))
    gr = np.stack([-R * Xt[:, 1], Xt[:, 1] * Xz[:, 0] - Xt[:, 0] * Xz[:, 1], R * Xt[:, 0]], 1)
    gr = np.asarray(_cartesian(gr, ph))
    Bgr = np.abs(np.sum(f["B"] * gr, 1)) / np.sqrt(B2 * np.sum(gr * gr, 1))
    rho = np.asarray(fields.force.rtz[:, 0])
    return {
        "du": float(np.linalg.norm(x[:n_u] - x0[:n_u]) / np.linalg.norm(x0[:n_u])),
        "dc_p": float(np.linalg.norm(x[n_u:] - x0[n_u:]) / np.linalg.norm(x0[n_u:])),
        "<p>": avg(f["p"]) / avg(ref["p"]),
        "<p^2>^1/2": np.sqrt(avg(f["p"] ** 2) / avg(ref["p"] ** 2)),
        "min p": float(f["p"].min() / ref["p"].max()),
        "rms J": rms(f["J"]) / rms(ref["J"]),
        "I_tor": I(f["J"]) / I(ref["J"]),
        "B-B_DESC": rms(f["B"] - np.asarray(fields.desc["B"])) / fields.rms["B"],
        "B.grad p": avg(np.abs(np.sum(f["B"] * gp, 1))) / avg(np.sqrt(B2 * np.sum(gp * gp, 1))),
        # B across DESC's surfaces: |B.grad rho| / |B| |grad rho|, all and rho > 0.9
        "B.grad rho": avg(Bgr),
        "B.grad rho edge": float(np.sum(w * Bgr * (rho > 0.9)) / np.sum(w * (rho > 0.9))),
        "<|F|>": avg(np.linalg.norm(f["F"], axis=1)) / fields.norm,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default=EQ, help="DESC equilibrium file (default: smooth-current precise_QA)")
    ap.add_argument("--coils", default=COILS, help="its coils (DESC file)")
    ap.add_argument("--frac", type=int, default=10)
    ap.add_argument("--refine", type=int, nargs="*", default=[],
                    help="continue on finer boxes, h = a / refine (e.g. --refine 20)")
    ap.add_argument("--kernel", default="truncated", choices=["hockney", "truncated"])
    ap.add_argument("--grid", default="cartesian", choices=["cartesian", "cylindrical"],
                    help="B_plasma solver: main's periodic box (with --kernel), or the "
                         "free-space solve on the cylindrical annulus (cylindrical_bs)")
    ap.add_argument("--margin", type=float, default=2.0,
                    help="cylindrical grid: walls this many h beyond the plasma's (R, Z) extent")
    ap.add_argument("--check-grid", default="cartesian", choices=["cartesian", "cylindrical"],
                    help="grid of the independent check (default the box: independent of "
                         "--grid cylindrical)")
    ap.add_argument("--free", default="all", choices=["current", "all"],
                    help="all: the current and the pressure (with W and pos held by default)")
    ap.add_argument("--maxiter", type=int, default=100, help="per box")
    ap.add_argument("--cg", type=int, default=100)
    ap.add_argument("--cg-tol", type=float, default=1e-2)
    ap.add_argument("--start", default=None,
                    help="npz with x (from a previous run) to start from instead of the fits")
    ap.add_argument("--ftol", type=float, default=2e-4,
                    help="stop when f decreases by less than this per iteration (over 5)")
    ap.add_argument("--precond", default="block", choices=["jacobi", "block"],
                    help="block: (|m|, |n|) blocks of the local columns (precondition.py)")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="seconds for the iterations of each box (leave time for the check)")
    ap.add_argument("--factors", type=float, nargs=3, default=(3, 3, 2.5))
    ap.add_argument("--colloc", type=float, default=2,
                    help="collocation resolution / resolution of F (potential + equilibrium L, M, N)")
    ap.add_argument("--check", type=float, default=3,
                    help="independent check grid resolution / resolution of F")
    ap.add_argument("--check-frac", type=int, default=None,
                    help="grid of the independent check, h = a / check-frac (default: 2 x the last "
                         "frac, at most a/20 unless the last is finer; axisymmetric with "
                         "--check-grid cylindrical: always 2 x the last)")
    ap.add_argument("--check-kernel", default="truncated", choices=["hockney", "truncated"])
    for k in ("force", "B", "J", "p", "W", "p2", "pos", "M"):
        ap.add_argument(f"--w-{k}", type=float, default=1.0 if k == "force" else None)
    ap.add_argument("--moments", nargs="+", default=["I", "R", "Z", "V"],
                    help="with --w-M: the current moments held (I_p, R_geo, Z_geo, V_eff/V)")
    ap.add_argument("--fix-I", action="store_true",
                    help="hold the enclosed toroidal current I(rho) (its zeta average) at the "
                         "start's: the a_l00 coefficients fixed")
    ap.add_argument("--p-scale", type=float, default=1.0,
                    help="pressure x this: the start's c_p and the stored-energy target (a beta scan)")
    ap.add_argument("--bext", default="field", choices=["field", "desc"],
                    help="B_ext at the collocation points: the given field, or B_DESC - "
                         "B_plasma(x0) on this box (x0 then has no box error in B)")
    ap.add_argument("--bplasma", default="box", choices=["box", "axisym"],
                    help="axisym: B_ext += B_plasma,accurate(x0) - B_plasma,box(x0), with the accurate "
                         "axisymmetric field of axisym_field.py (the box then only carries the "
                         "change from x0)")
    ap.add_argument("--correct-passes", type=int, default=0,
                    help="with --bplasma axisym: recompute the correction at the end state and "
                         "optimize again, this many times (defect correction)")
    ap.add_argument("--j-target-from", default=None,
                    help="w_J's target: J of this run's start (results/gs/NAME, in its own basis), the "
                         "same function for every basis (e.g. the x1 J-fit for a basis ladder)")
    ap.add_argument("--j-target", default="desc", choices=["desc", "start"],
                    help="target of the w_J term: J_DESC, or the start's J(x0) (a pure "
                         "regularizer: zero with zero gradient at x0)")
    ap.add_argument("--fit", default="T", choices=["T", "force", "J"],
                    help="start: the fit of T, or the force-matched fit of u (dense.py): "
                         "F_NEST(x0) - F_DESC minimized on the collocation points (with "
                         "--bext desc), or J fitted to J_DESC on the collocation points")
    ap.add_argument("--edge-power", type=int, default=2,
                    help="s of the potential's window (1 - rho^2)^s: J = O((1 - rho)^(s - 1)) "
                         "at the LCFS")
    ap.add_argument("--w-J-rho", type=float, default=None,
                    help="the w_J term only at rho < this (J free near the edge)")
    ap.add_argument("--fit-eps", type=float, default=1e-3,
                    help="weight of |J - J_DESC| in the force-matched fit")
    ap.add_argument("--dense", action="store_true",
                    help="B_plasma(u) as a precomputed matrix (dense.DenseFields): small "
                         "problems only (axisymmetric); the same result, much faster")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"],
                    help="precision of J and B on the box (and of the dense G): float32 limits "
                         "<|F|> / norm to ~1e-8")
    ap.add_argument("--cyl-dtype", default="float64", choices=["float32", "float64"],
                    help="precision of the cylindrical J -> B path (float32 is not accurate at "
                         "production sizes); --dtype sets main's box")
    ap.add_argument("--j-sampling", default=None, choices=["compact", "cached"],
                    help="J at the grid points from the flux grid: stencils formed per chunk "
                         "(compact, 25 bytes per point) or cached (1280 bytes per point); the same "
                         "values to rounding. Default: compact on the cylindrical grid, cached on the box")
    ap.add_argument("--S-dtype", default="float32", choices=["float32", "float64"],
                    help="storage of the cylindrical solver's single-layer matrices: float32 rounds S at "
                         "~6e-8 and floors B_plasma at ~1.2e-9 of rms B (gs_cyl_floor2.py); float64 removes it")
    ap.add_argument("--j-nodes", default="flux", choices=["flux", "direct"],
                    help="cylindrical grid: J at the annulus nodes interpolated from the flux grid, or "
                         "the basis evaluated at every node (node_current.py: no interpolation error)")
    ap.add_argument("--dense-g", default="box", choices=["box", "exact"],
                    help="with --dense: G from one box solve per column, or from the exact "
                         "axisymmetric field (axisym_field.plasma_field_matrix: no box error, "
                         "N = 0 only; the box is then only built, use a small --frac)")
    ap.add_argument("--dense-batch", type=int, default=8,
                    help="box solves at a time when forming the dense B_plasma (GPU memory)")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the independent check on a finer grid and box")
    ap.add_argument("--track", action="store_true",
                    help="drift of every iterate from the start (``drift``) in the history")
    ap.add_argument("--keep-iterates", action="store_true",
                    help="save every iterate x_k in the _iter.npz (xs), for small problems")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "opt"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    # with the pressure free: its volume average held and p >= 0 (``PressureEnergy``,
    # ``PressurePositivity``); without them F also drops by deflating the plasma
    for k in ("W", "pos"):
        if getattr(a, f"w_{k}") is None:
            setattr(a, f"w_{k}", 10.0 if a.free == "all" else 0.0)
    for k in ("B", "J", "p", "p2", "M"):
        if getattr(a, f"w_{k}") is None:
            setattr(a, f"w_{k}", 0.0)

    t0 = time.time()
    inp = DESCInput.load(a.eq, field=a.coils)
    # edge_power s: J = O((1 - rho)^(s - 1)) at the LCFS (passed only when not the default 2,
    # so that the cached fits keep their keys)
    cur = fit_cache.fit(BoundaryPotential, inp, factors=tuple(a.factors),
                        **({"edge_power": a.edge_power} if a.edge_power != 2 else {}))
    pre = fit_cache.fit(Pressure, inp, factors=tuple(a.factors), edge_order=2)
    sym = bool(inp.eq.sym)
    eq = inp.eq
    # F = J x B - grad p has about the resolution of the potential plus the equilibrium's
    b = cur.basis
    res = lambda f: tuple(int(round(f * (u + v))) for u, v in ((b.L, eq.L), (b.M, eq.M), (b.N, eq.N)))
    weights = {"force": a.w_force, "B": a.w_B, "J": a.w_J, "p": a.w_p, "W": a.w_W,
               "p2": a.w_p2, "pos": a.w_pos, "M": a.w_M}
    classes = {"force": ForceBalance, "B": FieldError, "J": CurrentError, "p": PressureError,
               "W": PressureEnergy, "p2": PressureSquare, "pos": PressurePositivity,
               "M": functools.partial(CurrentMoments, which=tuple(a.moments))}
    x = None
    if a.start is not None:  # its pressure rescaled from the scale it was run at
        st = np.load(a.start)
        x = jnp.asarray(st["x"])
        s0 = float(st["p_scale"]) if "p_scale" in st else 1.0
        if a.p_scale != s0:
            x = x.at[cur.n_free :].multiply(a.p_scale / s0)
    hist, m0, m1, xs, u_force = [], None, None, [], None
    for stage, frac in enumerate([a.frac] + list(a.refine)):
        box = make_box(inp, frac, a.grid, sym, a.margin)
        fr = ForceResidual(inp, box, resolution=res(a.colloc))
        fields = Fields(fr, cur, pre, kernel=a.kernel, dtype=getattr(jnp, a.dtype),
                        cyl_dtype=getattr(jnp, a.cyl_dtype), S_dtype=getattr(jnp, a.S_dtype),
                        **({} if a.j_sampling is None else {"j_compact": a.j_sampling == "compact"}))
        if a.j_nodes == "direct" and a.grid == "cylindrical":  # J at the annulus nodes from the basis
            from node_current import NodeFields
            fields = NodeFields.from_fields(fields, cur, box)
        if a.dense:  # B_plasma = G u + B_I; checked against the box solve along a random u
            v = jnp.asarray(np.random.default_rng(0).standard_normal(fields.n_u))
            B_box = np.asarray(fields.B_plasma(v))
            fields = DenseFields.from_fields(fields, cur, batch=a.dense_batch, method=a.dense_g)
            err = np.linalg.norm(np.asarray(fields.B_plasma(v)) - B_box) / np.linalg.norm(B_box)
            print(f"  dense B_plasma against the box solve, random u: {err:.1e}"
                  + (" (the box's own error: G is exact)" if a.dense_g == "exact" else ""), flush=True)
        if a.fit in ("force", "J"):  # x0 = (force-matched u or J fit, the pressure fit)
            if u_force is None:
                u_force, sc = force_matched_u(fr, cur, eps=a.fit_eps if a.fit == "force" else np.inf,
                                              desc={k: np.asarray(v) for k, v in fields.desc.items()})
                print(f"{a.fit}-fitted start: <|(J - J_DESC) x B_DESC|> / norm (= F_NEST - F_DESC "
                      f"at x0 with --bext desc) {sc['given fit']['F - F_DESC']:.2e} (fit of T) -> "
                      f"{sc['force fit']['F - F_DESC']:.2e}; rms J - J_DESC "
                      f"{sc['given fit']['J - J_DESC']:.2e} -> {sc['force fit']['J - J_DESC']:.2e}",
                      flush=True)
            fields.x0 = jnp.concatenate([jnp.asarray(u_force), fields.x0[fields.n_u :]])
        if a.bext == "desc":  # B(x0) = B_DESC at the points, in this box's discretization
            B_err = np.asarray(fields.compute(fields.x0)["B"]) - np.asarray(fields.desc["B"])
            fields.B_ext = fields.desc["B"] - fields.compute(fields.x0)["B_plasma"]
            w = np.asarray(fields.sw) ** 2
            print(f"B_ext = B_DESC - B_plasma(x0): replaces the field's, which missed B_DESC by "
                  f"{np.sqrt(np.sum(w * np.sum(B_err**2, 1))) / fields.rms['B']:.2e} (rms)",
                  flush=True)
        if a.bplasma == "axisym":  # the box's error in B_plasma(x0) removed (coils' field kept)
            B_acc = np.asarray(cylindrical_to_cartesian(plasma_field_cached(
                cur, np.asarray(fr.rtz), u=np.asarray(fields.x0[: fields.n_u])), np.asarray(fr.phi)))
            B_box = np.asarray(fields.compute(fields.x0)["B_plasma"])
            w = np.asarray(fields.sw) ** 2
            err = lambda v: np.sqrt(np.sum(w * np.sum(v**2, 1))) / fields.rms["B"]  # noqa: E731
            print(f"B_plasma(x0): box against accurate {err(B_box - B_acc):.2e}; B(x0) - B_DESC "
                  f"{err(np.asarray(fields.B_ext) + B_box - np.asarray(fields.desc['B'])):.2e} -> "
                  f"{err(np.asarray(fields.B_ext) + B_acc - np.asarray(fields.desc['B'])):.2e}",
                  flush=True)
            B_coil, B_corr = np.asarray(fields.B_ext), B_acc - B_box
            fields.B_ext = jnp.asarray(B_coil + B_corr)
        J_start = j_target(vars(a), fields, fields.x0, inp)
        if a.j_target_from:
            print(f"w_J target: J of {a.j_target_from}'s start (its own basis); against J(x0) here "
                  f"{float(jnp.linalg.norm(J_start - fields.compute(fields.x0)['J']) / jnp.linalg.norm(J_start)):.2e}",
                  flush=True)
        obj = ObjectiveFunction([(classes[k](w, scale=a.p_scale) if k in ("W", "p2") else
                                  classes[k](w, target=J_start, rho_max=a.w_J_rho) if k == "J" else
                                  classes[k](w))
                                 for k, w in weights.items() if w > 0], fields).build()
        free = np.ones(obj.dim_x, bool) if a.free == "all" else np.arange(obj.dim_x) < fields.n_u
        if a.fix_I:  # 2pi <T_theta>_(theta, zeta) = I h_0 + 2pi rho^2 f <a>: only a_l00 enter
            md = cur.basis.modes
            fix = cur.S.n_chi + np.nonzero((md[:, 1] == 0) & (md[:, 2] == 0))[0]
            free[fix] = False
        x0 = obj.x0
        if x is None:
            x = x0
            if a.p_scale != 1.0:  # the start's pressure scaled (once, before the first box)
                x = x.at[fields.n_u :].multiply(a.p_scale)
        if a.fix_I:  # DESC's I(rho) (the fits'), also from a start that changed it
            x = x.at[fix].set(x0[fix])
        print(f"{inp}\n[box {stage + 1}] h = a/{frac} {box.shape}, collocation {fr.tgrid.shape}, "
              f"unknowns {obj.dim_x:,} ({int(free.sum()):,} free), residual {obj.dim_f:,}; "
              f"objectives {weights}; "
              f"{'cylindrical annulus ' + repr(box) if a.grid == 'cylindrical' else 'kernel ' + a.kernel}; "
              f"setup {time.time() - t0:.0f} s; "
              f"{obj.dim_f / int(free.sum()):.1f} residuals per free unknown", flush=True)
        if m0 is None:
            m0 = fr.metrics(np.asarray(fields.compute(x0)["F"]))

        ref = {k: np.asarray(v) for k, v in fields.compute(x0).items()} if a.track else None
        d_start = None
        if a.track and not hist:
            d_start = drift(fields, x, x0, ref)
            print("drift " + json.dumps(d_start), flush=True)

        def save(x, h, frac=frac):  # every iteration, so that a time-out keeps the last x
            if a.track:
                if d_start is not None and "drift" not in h[0]:  # the start (iteration 0)
                    h[0]["drift"] = d_start
                h[-1]["drift"] = drift(fields, x, x0, ref)
                print("drift " + json.dumps(h[-1]["drift"]), flush=True)
            if a.keep_iterates:
                xs.append(np.asarray(x))
            np.savez(a.out + "_iter.npz", x=np.asarray(x), x0=np.asarray(x0), free=free,
                     p_scale=a.p_scale, **({"xs": np.array(xs)} if a.keep_iterates else {}),
                     **({"B_ext": np.asarray(fields.B_ext)} if a.bplasma == "axisym" else {}))
            with open(a.out + "_iter.json", "w") as f:
                json.dump({"args": vars(a), "box": frac, "history": hist + h}, f, indent=1)

        P = None
        if a.precond == "block":  # the local blocks at the start of this box
            P = BlockPreconditioner(obj, x, mode_groups(mode_labels(fields, cur, pre), free),
                                    structure=cur.S)
        x, h = levenberg_marquardt(obj, x0=x, free=free, maxiter=a.maxiter, cg_maxiter=a.cg,
                                   cg_tol=a.cg_tol, callback=save, time_limit=a.time_limit,
                                   precond=P, ftol=a.ftol)
        hist += [dict(e, box=frac) for e in h]
        for k in range(a.correct_passes if a.bplasma == "axisym" else 0):
            # defect correction: the box error removed again, now at the current x (the box only
            # carries the change from here), then the same optimization from x
            B_acc = np.asarray(cylindrical_to_cartesian(plasma_field_cached(
                cur, np.asarray(fr.rtz), u=np.asarray(x[: fields.n_u])), np.asarray(fr.phi)))
            B_box = np.asarray(fields.compute(x)["B_plasma"])
            w = np.asarray(fields.sw) ** 2
            err = lambda v: np.sqrt(np.sum(w * np.sum(v**2, 1))) / fields.rms["B"]  # noqa: E731
            F_old = obj.report(jnp.asarray(x))["<|F|>/norm"]
            fields.B_ext = jnp.asarray(B_coil + B_acc - B_box)
            obj = ObjectiveFunction(obj.objectives, fields).build()
            print(f"[correction pass {k + 1}] B_plasma(x): box against accurate {err(B_box - B_acc):.2e}; "
                  f"since the last correction {err(B_box + np.asarray(B_corr) - B_acc):.2e}; <|F|>/norm "
                  f"{F_old:.3e} -> {obj.report(jnp.asarray(x))['<|F|>/norm']:.3e} with it", flush=True)
            B_corr = B_acc - B_box
            if a.precond == "block":
                P = BlockPreconditioner(obj, x, mode_groups(mode_labels(fields, cur, pre), free),
                                        structure=cur.S)
            x, h = levenberg_marquardt(obj, x0=x, free=free, maxiter=a.maxiter, cg_maxiter=a.cg,
                                       cg_tol=a.cg_tol, callback=save, time_limit=a.time_limit,
                                       precond=P, ftol=a.ftol)
            hist += [dict(e, box=frac, correction_pass=k + 1) for e in h]
        m1 = fr.metrics(np.asarray(fields.compute(x)["F"]))
        print(f"force error <|F|> / norm by rho on box {stage + 1} (h = a/{frac}), start -> end:")
        for k in m0["bins"]:
            print(f"   {k:10s} {m0['bins'][k]:.2e} -> {m1['bins'][k]:.2e}")
        print(f"   all        {m0['mean']:.2e} -> {m1['mean']:.2e} (DESC {fr.F_desc_vol / fr.norm:.2e})",
              flush=True)
        desc_err = fr.F_desc_vol / fr.norm
        # free the device arrays of this box: the jitted functions keep the
        # objective (a static argument) in their caches
        del obj, fields, P, fr
        jax.clear_caches()
        gc.collect()
    u, c_p = x[: cur.n_free], x[cur.n_free :]
    # independent check: F on a finer collocation grid and with B_plasma from a
    # finer box (and the truncated kernel; always with the given field as B_ext):
    # the optimum can otherwise fit the discretization error of B on the
    # optimization box (J on the box as in the
    # objective: the same as evaluating it exactly there, scripts/stellarator/diagnose_opt.py)
    # saved first: the check is the most memory-hungry step
    np.savez(a.out + ".npz", x=np.asarray(x), x0=np.asarray(x0), u=np.asarray(u),
             c_p=np.asarray(c_p), free=free, p_scale=a.p_scale)
    out = None
    if not a.no_check:
        last = ([a.frac] + list(a.refine))[-1]
        # a/30 is beyond one A100 for the 3-D box; the axisymmetric cylindrical check (one plane)
        # is cheap at any h, so it is always 2x finer than the last run
        axi_check = a.check_grid == "cylindrical" and int(inp.eq.N) == 0
        cf = a.check_frac or (2 * last if axi_check else min(2 * last, max(last, 20)))
        cbox = make_box(inp, cf, a.check_grid, sym, a.margin)
        out, shape = check(inp, cbox, cur, pre, res(a.check), a.check_kernel,
                           {"start": x0, "end": x})
        print(f"check: grid {shape}, {a.check_grid} h = a/{cf} {cbox.shape}, "
              f"{'cylindrical' if a.check_grid == 'cylindrical' else 'kernel ' + a.check_kernel}; "
              f"<|F|> / norm by rho, start -> end:")
        for k in out["start"]["bins"]:
            print(f"   {k:10s} {out['start']['bins'][k]:.2e} -> {out['end']['bins'][k]:.2e}")
        print(f"   all        {out['start']['mean']:.2e} -> {out['end']['mean']:.2e}")
    np.savez(a.out + ".npz", x=np.asarray(x), x0=np.asarray(x0), u=np.asarray(u),
             c_p=np.asarray(c_p), free=free, p_scale=a.p_scale, rho_bins=list(m0["bins"]),
             before=list(m0["bins"].values()), after=list(m1["bins"].values()))
    with open(a.out + ".json", "w") as f:
        json.dump({"args": vars(a), "history": hist, "before": m0, "after": m1, "check": out,
                   "desc_force_error": desc_err}, f, indent=1)
    print(f"saved {a.out}.npz/.json; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
