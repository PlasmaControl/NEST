"""NEST with a free boundary: force balance and B.n = 0 on a moving boundary, solved together.

The unknowns are the current potential u, the pressure c_p and the normal
displacement xi of the boundary (``nest/free_boundary.py``). The objectives:

    ForceBalance            F = J x B - grad p                       (--w-force)
    TransportedCurrentError J - J_DESC, J_DESC carried with the boundary (--w-J)
    StoredEnergy            int p dV held at the start                (--w-W)
    PressurePositivity      p >= 0                                    (--w-pos)
    BoundaryNormalField     B.n / |B| on the boundary                 (--w-Bn)
    ToroidalFlux            Psi through phi = 0 held at the start     (--w-Psi)

Pressure balance [[B^2/2mu0 + p]] = 0 holds identically (no surface current, p = 0
on the boundary), so it needs no residual.

Anchors: on each box, Levenberg-Marquardt runs ``--inner`` iterations (or until
the boundary has moved by ``--max-disp``) with the geometry linearized about the
anchor where it is costly (B_ext at moved points, J on the box); then the
displacement is absorbed into the geometry (``Displacement.apply``), xi -> 0, and
everything is rebuilt. This repeats until the boundary moves by less than
``--disp-tol`` in an anchor, or ``--anchors`` times; ``--refine`` continues on
finer boxes. Every anchor saves x (u, c_p) and its geometry as a DESC file:

    <out>.npz       x, x0, history (as optimize_qa.py: --x of poincare.py)
    <out>_eq.h5     the displaced geometry (--eq of poincare.py and poincare_flux.py)
    <out>.json      history, per-anchor B.n, displacement, force error, check

The start is the fits of the DESC equilibrium (``--eq``) or a previous x (``--start``,
e.g. a fixed-boundary optimize_qa.py run on the same equilibrium; with ``--start-eq``
its geometry, to continue a free-boundary run).

Grid of B_plasma (``--grid``): main's periodic Cartesian box, or the cylindrical
annulus (``cylindrical_bs``; ``--N-phi`` planes per field period, walls ``--pad``
metres beyond the plasma so that the boundary can move within one grid per stage;
the annulus solver, built once per stage, is shared by its anchors, and J at the
inside nodes is interpolated with stencils formed on the fly).

run (GPU): sbatch --qos=gpu-stellar --time=06:00:00 run.sbatch scripts/optimize_fb.py \
    --eq desc_input/vmec_free_boundary/NCSX/NCSX_fb_L16M16N12.h5 \
    --coils desc_input/vmec_free_boundary/NCSX/NCSX_coils.h5 --w-J 1e-3 \
    --start results/ncsx/fixed.npz --refine 20 --out results/ncsx/fb
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from desc import set_device  # noqa: E402

set_device(os.environ.get("NEST_DEVICE", "cpu"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import desc.io  # noqa: E402
import fit_cache  # noqa: E402
from boundary_potential import BoundaryPotential  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc_input import DESCInput  # noqa: E402
from force import ForceResidual  # noqa: E402
from free_boundary import (BoundaryNormalField, Displacement, FreeBoundaryFields,  # noqa: E402
                           FreeBoundaryObjective, FreeBoundaryPreconditioner, StoredEnergy,
                           ToroidalFlux, TransportedCurrentError, TransportedCurrentMoments,
                           reference, release)
from cylindrical_grid import CylindricalGrid  # noqa: E402
from grid import CartesianGrid, plasma_extent  # noqa: E402
from objective import (BGradP, CurrentError, ForceBalance, ObjectiveFunction,  # noqa: E402
                       PressurePositivity)
from optimize import levenberg_marquardt  # noqa: E402
from precondition import BlockPreconditioner, mode_groups, mode_labels  # noqa: E402
from label_pressure import LabelDamping, LabelPressure  # noqa: E402
from pressure import Pressure  # noqa: E402

jax.config.update("jax_enable_x64", True)


def rebase(cur, pre, eq, x=None):
    """The fits' structure on the geometry of ``eq``, with x = (u, c_p) if given."""
    u, c = (cur.u, pre.c) if x is None else (np.asarray(x[: cur.n_free]),
                                             np.asarray(x[cur.n_free :]))
    return BoundaryPotential(cur.basis, u, cur.I, eq, cur.edge_power, cur.S), pre.with_c(c, eq)


def boundary_check(inp, box, cur, pre, x, kernel, n_theta=256, n_zeta=128, direct=(32, 2)):
    """B.n / |B| (rms over the area, max) on the boundary of ``inp`` for x = (u, c_p),
    on a grid over one field period: B_plasma by direct Biot-Savart (``DirectField``
    with ``direct`` = (n_rho, q); independent of any box) and, for comparison, from
    ``box``."""
    from free_boundary import DirectField
    from flux_interp import TensorGeometry, TensorGrid
    from fourier import fft_J
    from free_boundary import boundary_normal
    from magnetic_field import interpolate
    from objective import Fields, _cartesian
    from separable import Separable
    eq = inp.eq
    u = jnp.asarray(x[: cur.n_free])
    cyl = isinstance(box, CylindricalGrid)
    if not cyl:
        fr = ForceResidual(inp, box, rho_slice=slice(0, 1))
        fields = Fields(fr, cur, pre, kernel=kernel)
        B_box = np.asarray(fields._solve(fft_J(fields.J_box(u)), fields.axes, fields.K_hat))
    theta = 2 * np.pi * np.arange(n_theta) / n_theta
    zeta = 2 * np.pi * np.arange(n_zeta) / (n_zeta * eq.NFP)
    g = TensorGrid([1.0], theta, zeta, eq.NFP)
    geo = TensorGeometry(Separable([eq.R_basis, eq.Z_basis], [eq.R_lmn, eq.Z_lmn], eq.NFP), g)
    phi = g.nodes()[:, 2]
    R, Z = np.asarray(geo.R).ravel(), np.asarray(geo.Z).ravel()
    xb = np.column_stack([R * np.cos(phi), R * np.sin(phi), Z])
    B_ext = inp.B_ext(xb)
    n = np.asarray(_cartesian(boundary_normal(geo).reshape(-1, 3), phi))
    dA = np.linalg.norm(n, axis=1)
    out = {}
    df = DirectField(cur, Displacement(eq, 1, 0), (n_theta, n_zeta), direct[0], direct[1],
                     sym=False)
    fields_B = [("direct", np.asarray(df(u, cur.I, jnp.zeros(df.mg.A.shape[0]), jnp.asarray(xb))))]
    if not cyl:
        fields_B.append(("box", interpolate(B_box, fields.axes, xb)))
        del fields, B_box
    for name, B_pl in fields_B:
        B = B_ext + B_pl
        rel = np.sum(B * n, 1) / dA / np.linalg.norm(B, axis=1)
        tag = "" if name == "direct" else " (box)"
        out[f"Bn/|B| rms{tag}"] = float(np.sqrt(np.sum(dA * rel**2) / dA.sum()))
        out[f"Bn/|B| max{tag}"] = float(np.abs(rel).max())
    del df
    return out


def force_check_cyl(inp, grid, cur, pre, resolution, x, solver=None, n_rho=8, **kw):
    """``optimize_qa.check`` on a cylindrical grid (stencils on the fly, the annulus
    solver shared by the radial pieces): <|F|>_vol / norm overall and by radius."""
    from objective import Fields
    full = ForceResidual(inp, grid, resolution=resolution, rho_slice=slice(0, 1))
    n_r = resolution[0] // 2 + 1
    F, dV, rho = [], [], []
    base = None
    for i in range(0, n_r, n_rho):
        fr = ForceResidual(inp, grid, resolution=resolution, rho_slice=slice(i, i + n_rho))
        fields = Fields(fr, cur, pre, share=base, cyl_stencil=True, cyl_solver=solver, **kw)
        base = fields if base is None else base
        F.append(np.linalg.norm(np.asarray(fields.compute(jnp.asarray(x))["F"]), axis=1) / fr.norm)
        dV.append(fr.dV)
        rho.append(fr.rtz[:, 0])
        del fields, fr
        gc.collect()
    del base
    jax.clear_caches()
    gc.collect()
    f, dV, rho = np.concatenate(F), np.concatenate(dV), np.concatenate(rho)
    edges = (0.0, 0.5, 0.8, 0.9, 0.95, 1.0)
    avg = lambda m: float(np.sum(f * dV * m) / np.sum(dV * m))  # noqa: E731
    out = {"mean": avg(1.0), "bins": {f"{a:g}-{b:g}": avg((rho > a) & (rho <= b))
                                      for a, b in zip(edges[:-1], edges[1:])}}
    return out, (n_r,) + full.tgrid.shape[1:]


def annulus(inp, h, pad, N_phi=None):
    """The cylindrical grid around the plasma of ``inp``, walls pad [m] beyond its (R, Z)
    extent (z-symmetric), N_phi planes per period (default: arc spacing h)."""
    R0, R1, Z0, Z1 = plasma_extent(inp)
    Zm = max(abs(Z0), abs(Z1)) + pad
    return CylindricalGrid.around(inp, h, extent=(R0 - pad, R1 + pad, -Zm, Zm), N_phi=N_phi)


def free_solver(solver):
    """Delete the device arrays of a ``CylindricalSolver`` (the end of its stage)."""
    if solver is None:
        return
    for k in solver._dynamic:
        v = getattr(solver, k, None)
        if isinstance(v, jax.Array) and not v.is_deleted():
            v.delete()


def _peak():
    """", GPU peak x GB" when the allocator reports it (not the platform allocator)."""
    try:
        return f", GPU peak {jax.devices()[0].memory_stats()['peak_bytes_in_use'] / 1e9:.1f} GB"
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", required=True, help="DESC equilibrium (free boundary recommended)")
    ap.add_argument("--coils", required=True)
    ap.add_argument("--start", default=None, help="npz with x = (u, c_p) to start from")
    ap.add_argument("--start-eq", default=None,
                    help="the geometry of --start (a previous <out>_eq.h5); default --eq")
    ap.add_argument("--targets", default=None,
                    help="the .json of a previous run: hold its W, Psi and B.n normalization "
                         "(to continue it with --start, --start-eq)")
    ap.add_argument("--frac", type=int, default=10)
    ap.add_argument("--refine", type=int, nargs="*", default=[])
    ap.add_argument("--grid", default="cartesian", choices=["cartesian", "cylindrical"])
    ap.add_argument("--N-phi", type=int, nargs="*", default=[],
                    help="cylindrical: planes per field period (odd) of each stage (the last "
                         "repeated; default the arc spacing h)")
    ap.add_argument("--pad", type=float, default=0.02,
                    help="cylindrical: walls this far [m] beyond the plasma's (R, Z) extent")
    ap.add_argument("--mode-chunk", type=int, default=16,
                    help="cylindrical: toroidal modes per step of the annulus solve")
    ap.add_argument("--oversample", type=int, default=1,
                    help="cylindrical: J sampled on oversample x N_phi planes, the annulus solved "
                         "on N_phi with J's exact low toroidal modes (Fields cyl_oversample)")
    ap.add_argument("--interp-order", type=int, default=8,
                    help="cylindrical: B to the points by local Lagrange interpolation of this "
                         "order in (r, z) (0: the global Chebyshev interpolant)")
    ap.add_argument("--rblock", type=int, default=32,
                    help="cylindrical: J's FFT in phi this many rows of r at a time (cyl_rblock)")
    ap.add_argument("--xi-precond", default=None,
                    help="npz of the xi block G of the preconditioner (for xi in metres): loaded "
                         "if it exists (e.g. formed on a coarser grid), else formed and saved "
                         "(n_xi products J^T J e_k)")
    ap.add_argument("--check-N-phi", type=int, default=None,
                    help="cylindrical: planes of the check grid (default 1.5 x the last stage's)")
    ap.add_argument("--kernel", default="truncated", choices=["hockney", "truncated"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--factors", type=float, nargs=3, default=(3, 3, 2.5))
    ap.add_argument("--colloc", type=float, default=2)
    ap.add_argument("--colloc-rho", type=float, default=1.0,
                    help="this times more collocation radii (to resolve island-shaped p)")
    ap.add_argument("--pressure", default="fz", choices=["fz", "label"],
                    help="p as a Fourier-Zernike series, or p = P(chi) with a label field chi "
                         "(label_pressure.py: its surfaces can form islands)")
    ap.add_argument("--chi-M", type=int, default=16, help="label: poloidal modes of chi (in theta*)")
    ap.add_argument("--chi-N", type=int, default=4, help="label: toroidal modes of chi (x NFP)")
    ap.add_argument("--chi-rho0", type=float, default=0.15, help="label: chi = rho below this")
    ap.add_argument("--chi-knots", type=float, nargs=3, default=(0.04, 0.0125, 0.04),
                    metavar=("COARSE", "DENSE", "HALF"),
                    help="label: knot spacing, and DENSE within HALF of each rational of iota")
    ap.add_argument("--chi-m-max", type=int, default=14,
                    help="label: rationals iota = NFP n / m with m <= this get dense knots")
    ap.add_argument("--chi-filter", type=float, nargs=2, default=(8.0, 2.0), metavar=("M_C", "N_C"),
                    help="label: damping weight 1 + (m/M_C)^4 + (n/N_C)^4 of chi's coefficients")
    ap.add_argument("--w-chi", type=float, default=1e-3, help="label: weight of the damping of chi")
    ap.add_argument("--f-target", type=float, default=None,
                    help="hold <|F|>/norm at this fraction of DESC's: after each anchor --w-J is "
                         "scaled by sqrt(target / F) (in [1/2, 2]), the multiplier of the "
                         "constraint |F| <= target on min |J - J_target|^2); only when F has "
                         "stalled (less than 5%% down in an anchor) or is below the target")
    ap.add_argument("--w-J-range", type=float, default=30.0,
                    help="with --f-target: --w-J kept within its start value /x this")
    ap.add_argument("--xi-M", type=int, default=8, help="poloidal modes of the displacement")
    ap.add_argument("--xi-N", type=int, default=6, help="toroidal modes of the displacement")
    ap.add_argument("--anchors", type=int, default=6, help="re-anchorings per box, at most")
    ap.add_argument("--inner", type=int, default=20, help="iterations per anchor, at most")
    ap.add_argument("--max-disp", type=float, default=None,
                    help="re-anchor once the boundary moved this much [m] (default a/100)")
    ap.add_argument("--max-step", type=float, default=None,
                    help="largest motion of the boundary in one iteration [m] (default max-disp / 3)")
    ap.add_argument("--disp-tol", type=float, default=None,
                    help="stop a box when an anchor moves the boundary by less [m] (default a/3000)")
    ap.add_argument("--fix-I", action="store_true",
                    help="hold the enclosed toroidal current I(rho) (the a_l00 coefficients, "
                         "carried with the coordinates), as DESC holds the current profile")
    ap.add_argument("--edge-power", type=int, default=2,
                    help="s of the current's edge factor (1 - rho^2)^s: J = O((1 - rho)^(s - 1)) at "
                         "the LCFS (2: J -> 0 linearly; 1: J finite there, still no sheet current)")
    ap.add_argument("--fix-I-modes", type=int, default=None,
                    help="with --fix-I: hold only the K lowest radial modes a_l00 of I(rho) (default all)")
    ap.add_argument("--psi-scale", type=float, default=1.0,
                    help="hold the toroidal flux at this times its target (--targets) or start value")
    ap.add_argument("--fixed-boundary", action="store_true",
                    help="hold xi = 0 (the boundary fixed) and drop B.n from the residual (still "
                         "reported): the fixed-boundary problem with the same residual otherwise")
    ap.add_argument("--j-target", default="desc", choices=["desc", "start"],
                    help="target of the current term (--w-J): DESC's J, or the start fit's own J "
                         "(in the basis; DESC's is not where it does not vanish on the LCFS)")
    ap.add_argument("--direct", type=int, nargs=2, default=(32, 4), metavar=("N_RHO", "Q"),
                    help="B_plasma on the boundary by direct Biot-Savart: Gauss points in rho "
                         "and refinement q (even) of the boundary grid; 0 0 for the box's")
    ap.add_argument("--direct-tangent", type=int, nargs=2, default=(0, 0), metavar=("N_RHO", "Q"),
                    help="differentiate B on the boundary with a direct field of this resolution "
                         "(exact Jacobian rows of B.n, slower); 0 0: the box's derivative")
    ap.add_argument("--grad-every", type=int, default=5,
                    help="recompute the gradient of B_ext every this many anchors (else reuse it)")
    ap.add_argument("--cg", type=int, default=100)
    ap.add_argument("--cg-tol", type=float, default=1e-2)
    ap.add_argument("--ftol", type=float, default=None)
    ap.add_argument("--time-limit", type=float, default=None, help="seconds, for all boxes")
    for k, v in (("force", 1.0), ("J", 0.0), ("W", 10.0), ("pos", 10.0), ("Bn", 1.0),
                 ("Psi", 10.0), ("M", 0.0), ("Bp", 0.0)):
        ap.add_argument(f"--w-{k}", type=float, default=v)
    ap.add_argument("--moments", nargs="+", default=["I", "R", "Z", "V"],
                    help="with --w-M: the current moments held at the start fit's (I_p, R_geo, "
                         "Z_geo, V_eff/V; as the DSHAPE runs)")
    ap.add_argument("--check-frac", type=int, default=None,
                    help="box of the final check (default 2 x the last, at most a/20)")
    ap.add_argument("--check", type=float, default=3)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "fb"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t0 = time.time()
    dtype = getattr(jnp, a.dtype)

    inp0 = DESCInput.load(a.eq, field=a.coils)
    sym = bool(inp0.eq.sym)
    cur0 = fit_cache.fit(BoundaryPotential, inp0, factors=tuple(a.factors),
                         **({} if a.edge_power == 2 else {"edge_power": a.edge_power}))
    if a.pressure == "label":
        pre0 = LabelPressure.fit(inp0.eq, M=a.chi_M, N=a.chi_N, rho0=a.chi_rho0,
                                 coarse=a.chi_knots[0], dense=a.chi_knots[1], half=a.chi_knots[2],
                                 m_max=a.chi_m_max, filt=tuple(a.chi_filter))
        pre0.save(a.out + "_p.npz")  # its structure (c: x[n_u:] of <out>.npz)
    else:
        pre0 = fit_cache.fit(Pressure, inp0, factors=tuple(a.factors), edge_order=2)
    b, eq0 = cur0.basis, inp0.eq
    res = lambda f: tuple(int(round(f * s * (u + v))) for s, u, v in  # noqa: E731
                          ((a.colloc_rho, b.L, eq0.L), (1, b.M, eq0.M), (1, b.N, eq0.N)))
    ref = reference(inp0, res(a.colloc), current=cur0 if a.j_target == "start" else None)
    a0 = inp0.a
    max_disp = a.max_disp or a0 / 100
    disp_tol = a.disp_tol or a0 / 3000
    max_step = a.max_step or max_disp / 3
    x0 = np.concatenate([cur0.u, pre0.c])
    x = x0 if a.start is None else np.asarray(np.load(a.start)["x"])[: len(x0)]
    eq_k = eq0 if a.start_eq is None else desc.io.load(a.start_eq)
    print(f"{inp0}\nreference: DESC force error {ref['F_desc_vol'] / ref['norm']:.3e}; "
          f"setup {time.time() - t0:.0f} s", flush=True)

    targets, hist, anchors_log, xi_scale, lam = {}, [], [], 1.0, 1e-2
    grad, grad_age = None, 0  # the gradient of B_ext, reused for a few anchors
    if a.targets is not None:
        with open(a.targets) as f:
            targets = dict(json.load(f)["targets"])
        print(f"targets of {a.targets}: {targets}", flush=True)
    deadline = None if a.time_limit is None else t0 + a.time_limit
    stop_all = False
    cyl = a.grid == "cylindrical"
    solver = grid0 = None
    for stage, frac in enumerate([a.frac] + list(a.refine)):
        P_block, P_xi = None, None
        free_solver(solver)
        solver = None
        n_phi = (a.N_phi[min(stage, len(a.N_phi) - 1)] if a.N_phi else None)
        for k in range(a.anchors):
            if deadline is not None and time.time() > deadline:
                stop_all = True
                break
            t1 = time.time()
            inp = DESCInput(eq_k, inp0.field, name=f"anchor {stage}.{k}", check=False)
            # one box per stage: a new box would change B_plasma by its discretization
            # error at every anchor (the factor-2 padding leaves room for the motion)
            if cyl:  # one annulus per stage (walls --pad beyond the stage's first plasma)
                if k == 0:
                    grid0 = annulus(inp, a0 / frac, a.pad,
                                    None if n_phi is None else n_phi * a.oversample)
                box = CylindricalGrid(grid0.r_axis, grid0.z_axis, grid0.NFP, grid0.N_phi)
                box.locate_plasma_planes(inp, sym=sym, verbose=k == 0)
                print(f"  {box}: {len(box.inside):,} inside nodes; located in "
                      f"{time.time() - t1:.0f} s", flush=True)
            else:
                if k == 0:
                    axes = CartesianGrid.periodic_box(inp, a0 / frac, odd=sym).axes
                box = CartesianGrid(axes, periodic=True).locate_plasma(inp)
            fr = ForceResidual(inp, box, resolution=res(a.colloc))
            fr.norm, fr.F_desc_vol = ref["norm"], ref["F_desc_vol"]
            cur, pre = rebase(cur0, pre0, eq_k, x)
            disp = Displacement(eq_k, a.xi_M, a.xi_N)
            reuse = grad is not None and grad_age < a.grad_every
            cyl_kw = (dict(cyl_stencil=True, cyl_solver=solver, mode_chunk=a.mode_chunk,
                           cyl_oversample=a.oversample, cyl_fused=True, cyl_rblock=a.rblock,
                           cyl_order=a.interp_order or None)
                      if cyl else {})
            fields = FreeBoundaryFields(fr, cur, pre, disp, ref, kernel=a.kernel, dtype=dtype,
                                        grad_B_ext=grad if reuse else None,
                                        direct=tuple(a.direct) if a.direct[0] > 0 else None,
                                        direct_tangent=(tuple(a.direct_tangent)
                                                        if a.direct_tangent[0] > 0 else False),
                                        **cyl_kw)
            if cyl:
                solver = fields.solver
            grad_age = grad_age + 1 if reuse else 1
            grad = np.asarray(fields.grad_B_ext)
            objs = [ForceBalance(a.w_force)]
            if a.w_J > 0:
                objs.append(TransportedCurrentError(a.w_J))
            if a.w_M > 0:
                tm = targets.get("M")
                objs.append(TransportedCurrentMoments(a.w_M, which=tuple(a.moments),
                                                      target=None if tm is None else np.asarray(tm)))
            if a.w_Bp > 0:
                objs.append(BGradP(a.w_Bp))
            if a.w_W > 0:
                objs.append(StoredEnergy(a.w_W, target=targets.get("W")))
            if a.w_pos > 0:
                objs.append(PressurePositivity(a.w_pos))
            if a.pressure == "label" and a.w_chi > 0:
                objs.append(LabelDamping(a.w_chi, pre))
            objs.append(BoundaryNormalField(0.0 if a.fixed_boundary else a.w_Bn,
                                            normalization=targets.get("B")))
            if a.w_Psi > 0:
                psi_t = targets.get("Psi")
                if a.psi_scale != 1.0 and psi_t is not None and not targets.get("Psi scaled"):
                    targets["Psi"] = psi_t = psi_t * a.psi_scale  # once: saved with the targets
                    targets["Psi scaled"] = a.psi_scale
                objs.append(ToroidalFlux(a.w_Psi, target=psi_t))
            obj = FreeBoundaryObjective(objs, fields).build()
            for o in objs:  # held at the very start, across anchors and boxes
                if o.name == "moments":
                    targets.setdefault("M", np.asarray(o.target).tolist())
                key = {"W": "W", "Psi": "Psi", "B.n": "B"}.get(o.name)
                if key == "B":
                    targets.setdefault(key, o.normalization)
                elif key is not None:
                    targets.setdefault(key, float(o.target))
            xf = jnp.concatenate([jnp.asarray(x), jnp.zeros(fields.n_xi)])
            n_up = fields.n_u + fields.n_p
            free = np.ones(obj.dim_x, bool)
            if a.fixed_boundary:
                free[n_up:] = False
            if a.fix_I:  # 2 pi <T_theta>_(theta, zeta) = I h_0 + 2 pi rho^2 f <a>: a_l00
                md = cur.basis.modes
                l00 = np.nonzero((md[:, 1] == 0) & (md[:, 2] == 0))[0]
                if a.fix_I_modes is not None:  # the K lowest radial modes only
                    l00 = l00[np.argsort(md[l00, 0], kind="stable")][: a.fix_I_modes]
                free[cur.S.n_chi + l00] = False
            if P_block is None:  # (u, c_p) blocks on the first anchor of each box
                pv = fields.fixed_view()
                pobjs = [ForceBalance(a.w_force, normalization=ref["norm"])]
                if a.w_J > 0:
                    pobjs.append(CurrentError(a.w_J, target=0.0, normalization=ref["rms"]["J"]))
                pobj = ObjectiveFunction(pobjs, pv).build()
                groups = mode_groups(mode_labels(pv, cur, pre), free[:n_up])
                P_block = BlockPreconditioner(pobj, xf[:n_up], groups, structure=cur.S)
                del pobj, pv
                dg = np.asarray(P_block.diag)
                iu, iq = np.arange(cur.n_free), cur.n_free + np.arange(getattr(pre, "n_q", pre.n))
                iu, iq = iu[free[iu]], iq[free[iq]]
                msg = f"  mean diag(J^T J): u {dg[iu].mean():.3e}, p {dg[iq].mean():.3e}"
                if a.pressure == "label":
                    idd = cur.n_free + pre.n_q + np.arange(pre.n_d)
                    msg += f", chi {dg[idd].mean():.3e}"
                    if "chi_scale" not in targets:  # the columns of d to the size of u's
                        s = float(np.sqrt(dg[iu].mean() / dg[idd].mean()))
                        targets["chi_scale"] = pre0.chi_scale = pre.chi_scale = s
                        fields.p_col.chi_scale = jnp.asarray(s)
                        P_block.rescale(idd, s)
                        for o in objs:
                            if isinstance(o, LabelDamping):
                                o._w = o._w * s
                        pre0.save(a.out + "_p.npz")
                        msg += f" -> chi unknowns scaled by {s:.3e}"
                print(msg, flush=True)
            if P_xi is None:
                G = None
                if a.xi_precond and os.path.exists(a.xi_precond):
                    G = np.load(a.xi_precond)["G"]
                    print(f"  xi block from {a.xi_precond}", flush=True)
                P_xi = FreeBoundaryPreconditioner(P_block, obj, xf, G=G)
                if a.xi_precond and G is None:  # G of xi in metres (fields.xi_scale is 1 here)
                    np.savez(a.xi_precond, G=P_xi.G)
                # xi / s as the unknowns, its columns the size of the current's
                xi_scale = P_xi.scale()
                P_xi.rescale(xi_scale)
                print(f"  unknowns xi / {xi_scale:.2e} m", flush=True)
            fields.scale_xi(xi_scale)
            P = P_xi
            # the largest boundary displacement of xi, on a fine boundary grid
            g = LinearGrid(rho=1.0, M=2 * eq_k.M, N=2 * eq_k.N, NFP=eq_k.NFP)
            Fb = np.asarray(disp.basis.evaluate(g))
            dmax = lambda xf: float(np.abs(Fb @ fields.xi_metres(xf)).max())  # noqa: E731
            r0 = obj.report(xf)
            print(f"[{'grid' if cyl else 'box'} {stage + 1}, anchor {k}] h = a/{frac} {box.shape}, collocation "
                  f"{fr.tgrid.shape}, unknowns {obj.dim_x:,} ({fields.n_xi} boundary), residual "
                  f"{obj.dim_f:,}; B.n/|B| rms {r0['Bn/|B| rms']:.3e}, <|F|>/norm "
                  f"{r0['<|F|>/norm']:.3e}; setup {time.time() - t1:.0f} s", flush=True)

            def save(xf, h, frac=frac, stage=stage, k=k):
                d = dmax(xf)
                h[-1]["max |D|"] = d
                np.savez(a.out + "_iter.npz", x=np.asarray(xf[:n_up]), xi=fields.xi_metres(xf),
                         x0=x0)
                with open(a.out + "_iter.json", "w") as f:
                    json.dump({"args": vars(a), "box": frac, "anchor": k,
                               "history": hist + [dict(e, box=frac, anchor=k) for e in h]}, f,
                              indent=1)
                return d > max_disp

            left = None if deadline is None else max(deadline - time.time(), 60)
            # lam carried over from the last anchor (a re-anchoring changes the problem little)
            # fixed boundary: one anchor, nothing to re-anchor
            inner = a.inner * a.anchors if a.fixed_boundary else a.inner
            xf, h = levenberg_marquardt(obj, x0=xf, free=free, lam=lam, maxiter=inner,
                                        cg_maxiter=a.cg,
                                        cg_tol=a.cg_tol, callback=save, time_limit=left,
                                        precond=P, ftol=a.ftol, quadratic=False,
                                        step_limit=lambda d: max_step / max(dmax(d), 1e-30))
            hist += [dict(e, box=frac, anchor=k) for e in h]
            lam = max(h[-1]["lam"], 1e-6)
            r1 = obj.report(xf)
            d = dmax(xf)
            fe = fields.compute(xf)
            m = fr.metrics(np.asarray(fe["F"]))
            sw2 = np.asarray(fields.sw) ** 2
            B, gp = np.asarray(fe["B"]), np.asarray(fe["grad_p"])
            bgp = float(np.sum(sw2 * np.abs(np.sum(B * gp, 1)) / np.linalg.norm(B, axis=1))) / fields.norm
            W_now = float(np.asarray(fe["W"])[0])
            targets.setdefault("W0", float(np.asarray(fields.compute(fields.x0)["W"])[0]))
            r1["<|B.grad p|/|B|>/norm"], r1["W/W0"] = bgp, W_now / targets["W0"]
            anchors_log.append({"box": frac, "anchor": k, "iterations": len(h) - 1,
                                "max |D| [m]": d, "start": r0, "end": r1, "F bins": m["bins"],
                                "w_J": a.w_J})
            on_target = True
            if a.f_target is not None:  # the constraint |F| <= f_target F_DESC: its multiplier
                tau = a.f_target * ref["F_desc_vol"] / ref["norm"]
                F_now, F_was = r1["<|F|>/norm"], r0["<|F|>/norm"]
                on_target = abs(F_now / tau - 1) < 0.1
                w_J0 = targets.setdefault("w_J0", a.w_J)
                # only once this weight has done what it can (F stalled), or F is below
                # the target: else the anchor was just not converged
                if F_now < 0.9 * tau or F_now > 0.95 * F_was:
                    a.w_J = float(np.clip(a.w_J * np.clip(np.sqrt(tau / F_now), 0.5, 2.0),
                                          w_J0 / a.w_J_range, w_J0 * a.w_J_range))
                print(f"  <|F|>/norm {F_now:.3e} (anchor start {F_was:.3e}), target {tau:.3e}: "
                      f"w_J {a.w_J:.3e}", flush=True)
            print(f"  anchor {k}: {len(h) - 1} iterations, boundary moved by up to {d * 1e3:.3f} "
                  f"mm; B.n/|B| rms {r0['Bn/|B| rms']:.3e} -> {r1['Bn/|B| rms']:.3e} (max "
                  f"{r1['Bn/|B| max']:.2e}); <|F|>/norm {r0['<|F|>/norm']:.3e} -> "
                  f"{r1['<|F|>/norm']:.3e}; B.grad p/norm {bgp:.3e}; W/W0 {W_now / targets['W0']:.4f}; "
                  f"{time.time() - t1:.0f} s{_peak()}", flush=True)
            # absorb the displacement into the geometry
            x = np.asarray(xf[:n_up])
            eq_k = disp.apply(fields.xi_metres(xf))
            eq_k.save(a.out + "_eq.h5")
            np.savez(a.out + ".npz", x=x, x0=x0, p_scale=1.0, pressure=a.pressure)
            with open(a.out + ".json", "w") as f:
                json.dump({"args": vars(a), "anchors": anchors_log, "history": hist,
                           "targets": targets, "desc_force_error": ref["F_desc_vol"] / ref["norm"]},
                          f, indent=1)
            # this anchor's device arrays (JAX's caches would keep them: ``release``)
            release(obj, fields, fr, box)
            del obj, fields, fr, box, P
            jax.clear_caches()
            gc.collect()
            live = jax.live_arrays()
            print(f"  live device arrays after the anchor: {len(live)}, "
                  f"{sum(x.nbytes for x in live) / 1e9:.2f} GB", flush=True)
            if (d < disp_tol and on_target) or a.fixed_boundary:
                break
        if stop_all:
            break

    # independent check on the final geometry: F on a finer grid and box, B.n there
    check = None
    if not a.no_check:
        from optimize_qa import check as force_check
        inp = DESCInput(eq_k, inp0.field, name="final", check=False)
        last = ([a.frac] + list(a.refine))[-1]
        cur, pre = rebase(cur0, pre0, eq_k, x)
        if cyl:  # the last stage's (R, Z) spacing, 1.5 x its planes (another annulus)
            free_solver(solver)
            solver = None
            cf = a.check_frac or last
            n_last = grid0.N_phi // a.oversample
            n_chk = a.check_N_phi or int(1.5 * n_last) | 1
            cbox = annulus(inp, a0 / cf, a.pad, n_chk * a.oversample)
            cbox.locate_plasma_planes(inp, sym=sym)
            out, shape = force_check_cyl(inp, cbox, cur, pre, res(a.check), x,
                                         mode_chunk=a.mode_chunk, cyl_oversample=a.oversample,
                                         cyl_fused=True, cyl_rblock=a.rblock,
                                         cyl_order=a.interp_order or None)
            out = {"end": out}
            tag = f"annulus h = a/{cf}, N_phi {n_chk}"
        else:
            cf = a.check_frac or min(2 * last, max(last, 20))
            cbox = CartesianGrid.periodic_box(inp, a0 / cf, odd=sym).locate_plasma(inp)
            out, shape = force_check(inp, cbox, cur, pre, res(a.check), a.kernel,
                                     {"end": jnp.asarray(x)})
            tag = f"box h = a/{cf}"
        bn = boundary_check(inp, cbox, cur, pre, x, a.kernel)
        check = {"force": out["end"], "boundary": bn, "grid": shape, "box": cf}
        print(f"check: grid {shape}, {tag}: <|F|>/norm {out['end']['mean']:.3e} "
              f"(DESC {ref['F_desc_vol'] / ref['norm']:.3e}); by rho " +
              ", ".join(f"{k} {v:.2e}" for k, v in out["end"]["bins"].items()) +
              f"; B.n/|B| rms {bn['Bn/|B| rms']:.3e}, max {bn['Bn/|B| max']:.3e} (direct"
              + (f"; box {bn['Bn/|B| rms (box)']:.3e})" if "Bn/|B| rms (box)" in bn else ")"),
              flush=True)
    with open(a.out + ".json", "w") as f:
        json.dump({"args": vars(a), "anchors": anchors_log, "history": hist, "targets": targets,
                   "check": check, "desc_force_error": ref["F_desc_vol"] / ref["norm"]}, f,
                  indent=1)
    print(f"saved {a.out}.npz, {a.out}_eq.h5, {a.out}.json; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
