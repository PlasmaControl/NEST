"""Boundary-adapted current potential (``BoundaryPotential``, J_potential_notes.pdf).

For each equilibrium, the potential fitted to DESC (T itself) and random
admissible coefficient vectors u are checked for:
  A. div J = 0, independently of the curl identity: sqrt(g) J^i = sqrt(g) J . e^i
     from the evaluated J and DESC's own e^i, sqrt(g), then central differences
  B. J = 0 on the LCFS (and its decay near it), for the fit and random u
  C. axis regularity: J at rho -> 0 bounded, independent of theta in the limit
     (spread ~ rho), and the exact rho = 0 value (the limit d_rho N / d_rho sqrt(g))
     equal to it; for random u restricted to |m| = 0, 1, 2 and for the fit
  D. net toroidal current int int sqrt(g) J^zeta drho dtheta = I on several
     cross-sections zeta = const, for the fit and random u
  E. DESC reconstruction: the fit error in T (with the dropped gauge alpha(zeta))
     and in J (components and |J|), by radius
  F. differentiability: JAX derivatives of J with respect to u vs finite
     differences (J is linear in u)
  G. save / load: same u, I, J, net current and J = 0 on the LCFS

run: pytest tests/test_boundary_potential.py -s     or     python tests/test_boundary_potential.py
"""

import os
import sys
import tempfile

import numpy as np

try:  # pytest is optional: the file also runs as a plain script
    import pytest
except ImportError:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nest"))

from desc import set_device  # noqa: E402

# DESC uses the CPU unless set_device('gpu') runs before any other DESC import
set_device(os.environ.get("NEST_DEVICE", "cpu"))

import desc.io  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from desc.basis import fourier  # noqa: E402
from desc.grid import Grid  # noqa: E402

from boundary_potential import BoundaryPotential, target_potential  # noqa: E402
from desc_grid import flux_grid  # noqa: E402

jax.config.update("jax_enable_x64", True)

EQ_FILES = [
    os.path.join(ROOT, "desc_input", "precise_QA_free_boundary_smooth_J",
                 "precise_QA_free_boundary_smooth_J_edgeJ_r0.3.h5"),
    os.path.join(ROOT, "desc_input", "W7-X_output.h5"),
]
TOL_DIV = 1e-5  # |sum_i d_i (sqrt(g) J^i)| / sum_i |d_i (sqrt(g) J^i)|: finite-difference
# truncation (random u at full resolution reach 2e-6); a real divergence is O(1)
TOL_EDGE = 1e-12  # max |J(rho = 1)| / max |J|
TOL_AXIS = 1e-4  # theta spread at rho = 1e-6, and |J(1e-6) - J(0)|, / max |J|
TOL_CURRENT = 1e-10  # |I_cross-section - I| / |I|
TOL_GRAD = 1e-8  # JAX vs finite-difference derivative, relative
_CACHE = {}


def load(path):
    if path not in _CACHE:
        eq = desc.io.load(path)
        eq = eq[-1] if hasattr(eq, "__len__") else eq
        _CACHE[path] = (eq, BoundaryPotential.fit(eq))
    return _CACHE[path]


def random_potentials(pot, n=3, seed=0, m_only=None):
    """Random admissible u (normal, scaled like the fit), optionally only |m| = m_only."""
    rng = np.random.default_rng(seed)
    scale = np.sqrt(np.mean(pot.u**2))
    keep = np.ones(pot.n_free, bool)
    if m_only is not None:
        m = np.concatenate([pot.chi_modes[:, 0], pot.basis.modes[:, 1], pot.basis.modes[:, 1]])
        keep = np.abs(m) == m_only
    return [pot.with_u(scale * rng.normal(size=pot.n_free) * keep) for _ in range(n)]


def surface(eq, rho, n_theta=64, n_zeta=16):
    t, z = np.meshgrid(np.linspace(0, 2 * np.pi, n_theta, endpoint=False),
                       np.linspace(0, 2 * np.pi / eq.NFP, n_zeta, endpoint=False), indexing="ij")
    return np.column_stack([np.full(t.size, rho), t.ravel(), z.ravel()])


def cross_section_current(eq, pot, zeta, n_rho=64, n_theta=None):
    """int int sqrt(g) J^zeta drho dtheta at zeta, sqrt(g) J^zeta = sqrt(g) J_phi / R."""
    n_theta = 2 * pot.basis.M + 9 if n_theta is None else n_theta
    x, w = np.polynomial.legendre.leggauss(n_rho)
    r, t = np.meshgrid((x + 1) / 2, np.linspace(0, 2 * np.pi, n_theta, endpoint=False),
                       indexing="ij")
    nodes = np.column_stack([r.ravel(), t.ravel(), np.full(r.size, zeta)])
    d = eq.compute(["R", "sqrt(g)"], grid=Grid(nodes, sort=False))
    sJz = np.asarray(d["sqrt(g)"]) * pot.evaluate(nodes)[:, 1] / np.asarray(d["R"])
    return np.sum(sJz * np.repeat(w / 2, n_theta)) * 2 * np.pi / n_theta


def check_divergence(eq, pot, n=100, delta=1e-5, seed=1):
    """A: sum_i d_i (sqrt(g) J^i) by central differences of sqrt(g) J . e^i (error
    ~ delta^2: 5e-7 at 1e-4, 5e-8 at 1e-5, then round-off)."""
    rng = np.random.default_rng(seed)
    x0 = np.column_stack([rng.uniform(0.05, 0.95, n), rng.uniform(0, 2 * np.pi, n),
                          rng.uniform(0, 2 * np.pi, n)])
    terms = []
    for i, key in enumerate(["e^rho", "e^theta", "e^zeta"]):
        vals = []
        for sgn in (1, -1):
            x = x0.copy()
            x[:, i] += sgn * delta
            d = eq.compute([key, "sqrt(g)"], grid=Grid(x, sort=False))
            vals.append(np.asarray(d["sqrt(g)"]) * np.sum(pot.evaluate(x) * np.asarray(d[key]), 1))
        terms.append((vals[0] - vals[1]) / (2 * delta))
    terms = np.array(terms)
    return np.max(np.abs(terms.sum(0)) / np.abs(terms).sum(0))


def check_boundary_potential(path):
    eq, pot = load(path)
    b = pot.basis
    print(f"\n{os.path.basename(path)}: a, b series (L,M,N)=({b.L},{b.M},{b.N}), "
          f"{pot.n_free} free coefficients ({pot.S.n_chi} chi + 2 x {b.num_modes}), "
          f"s = {pot.edge_power}, I = {pot.I:.6f} A")
    g = flux_grid(eq, 2 * eq.L, 2 * eq.M, 2 * eq.N)
    rnd = random_potentials(pot)

    # A. divergence
    div = [check_divergence(eq, p) for p in [pot] + rnd[:1]]
    print(f"A. div J by finite differences, |sum_i d_i(sqrt(g) J^i)| / sum_i |...|: "
          f"fit {div[0]:.1e}, random {div[1]:.1e}")

    # B. LCFS
    edge = []
    for p in [pot] + rnd:
        J_max = np.abs(p.evaluate(g)).max()
        edge.append([np.abs(p.evaluate(surface(eq, r))).max() / J_max for r in (1.0, 0.999, 0.99)])
    edge = np.array(edge)
    layer = [np.abs(pot.evaluate(surface(eq, r))).max() / np.abs(pot.evaluate(g)).max()
             for r in (0.9, 0.95, 0.98, 0.99, 0.999)]
    print(f"B. max |J| / max |J| on rho = 1: fit {edge[0, 0]:.1e}, random "
          f"{edge[1:, 0].max():.1e}; fit at rho = 0.9, 0.95, 0.98, 0.99, 0.999: "
          + ", ".join(f"{v:.3f}" for v in layer))

    # C. axis
    rhos = [1e-1, 1e-2, 1e-4, 1e-6, 0.0]
    axis = []
    for label, p in [("fit", pot)] + [(f"|m| = {m}", random_potentials(pot, 1, m, m)[0])
                                      for m in (0, 1, 2)]:
        J_max = np.abs(p.evaluate(g)).max()
        Js = [p.evaluate(surface(eq, r, 32, 4)).reshape(32, 4, 3) for r in rhos]
        spread = [np.abs(J - J.mean(0)).max() / J_max for J in Js]
        bounded = max(np.abs(J).max() for J in Js) / J_max
        lim = np.abs(Js[3] - Js[4]).max() / J_max
        axis.append((label, bounded, spread, lim))
        print(f"C. axis, {label}: max |J(rho <= 0.1)| / max |J| = {bounded:.2f}; theta spread / "
              f"max |J| at rho = " + ", ".join(f"{r:g}: {s:.1e}" for r, s in zip(rhos, spread))
              + f"; |J(1e-6) - J(0)| / max |J| = {lim:.1e}")

    # D. net current
    zetas = np.array([0.0, 0.37, 1.1]) * 2 * np.pi / eq.NFP
    I_err = max(abs(cross_section_current(eq, p, z) - p.I) / abs(p.I)
                for p in [pot] + rnd[:1] for z in zetas)
    print(f"D. net toroidal current through zeta = {np.round(zetas, 3).tolist()}: "
          f"max |I_cs - I| / |I| = {I_err:.1e} (fit and random)")

    # E. DESC reconstruction
    rho, theta, zeta, T_t = target_potential(eq, 2 * eq.L + 3, 2 * eq.M + 5, 2 * eq.N + 3)
    R_, Th, Ze = np.meshgrid(rho, theta, zeta, indexing="ij")
    nodes = np.column_stack([R_.ravel(), Th.ravel(), Ze.ravel()])
    T_t = T_t.reshape(-1, 2)
    T_t[:, 1] -= pot.gauge(nodes[:, 2])  # the dropped gauge
    dT = np.abs(pot.T(nodes) - T_t).max(0) / np.abs(T_t).max(0)
    J, J_desc = pot.evaluate(g), np.asarray(eq.compute("J", grid=g)["J"])
    r, J_max = np.asarray(g.nodes[:, 0]), np.abs(J_desc).max()
    mod = np.abs(np.linalg.norm(J, axis=1) - np.linalg.norm(J_desc, axis=1)) / J_max
    print(f"E. fit of T: max |T - T_desc| / max |T_desc|: T_theta {dT[0]:.1e}, T_zeta {dT[1]:.1e}")
    for r0 in (0.5, 0.8, 0.9, 1.0):
        k = r <= r0
        e = np.abs(J - J_desc)[k].max(0) / J_max
        print(f"   J vs DESC, rho <= {r0}: (J_R, J_phi, J_Z) {e[0]:.1e}, {e[1]:.1e}, {e[2]:.1e}; "
              f"|J| {mod[k].max():.1e}")

    # F. derivatives with respect to u
    f = pot.current_function()
    pts = [jnp.asarray(v) for v in surface(eq, 0.6, 4, 2).T]
    du = np.random.default_rng(5).normal(size=pot.n_free) * np.sqrt(np.mean(pot.u**2))
    u = jnp.asarray(pot.u)
    jvp = np.asarray(jax.jvp(lambda v: f(v, *pts), (u,), (jnp.asarray(du),))[1])
    fd = (np.asarray(f(u + du, *pts)) - np.asarray(f(u - du, *pts))) / 2
    grad = np.asarray(jax.grad(lambda v: f(v, *pts)[:, 1].sum())(u))
    k = np.argmax(np.abs(grad))
    e_k = np.zeros(pot.n_free)
    e_k[k] = 1.0
    fd_k = (np.asarray(f(u + e_k, *pts))[:, 1].sum() - np.asarray(f(u - e_k, *pts))[:, 1].sum()) / 2
    e_jvp = np.abs(jvp - fd).max() / np.abs(fd).max()
    e_grad = abs(grad[k] - fd_k) / abs(fd_k)
    print(f"F. dJ/du: jvp vs finite difference {e_jvp:.1e}; grad entry {k} vs finite "
          f"difference {e_grad:.1e}")

    # G. save / load
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, "pot.npz")
        pot.save(fn)
        pot2 = BoundaryPotential.load(fn, eq)
    same = (np.array_equal(pot2.u, pot.u) and pot2.I == pot.I
            and np.array_equal(pot2.evaluate(g), J))
    e2 = np.abs(pot2.evaluate(surface(eq, 1.0))).max() / np.abs(J).max()
    I2 = abs(cross_section_current(eq, pot2, 0.0) - pot.I) / abs(pot.I)
    print(f"G. save / load: u, I and J identical: {same}; J on rho = 1 {e2:.1e}; "
          f"net current {I2:.1e}")

    assert max(div) < TOL_DIV
    assert edge[:, 0].max() < TOL_EDGE
    for _, bounded, spread, lim in axis:
        assert bounded < 10 and spread[3] < TOL_AXIS and spread[4] < TOL_AXIS and lim < TOL_AXIS
    assert I_err < TOL_CURRENT
    assert e_jvp < TOL_GRAD and e_grad < TOL_GRAD
    assert same and e2 < TOL_EDGE and I2 < TOL_CURRENT


try:
    @pytest.mark.parametrize("path", EQ_FILES, ids=[os.path.basename(f) for f in EQ_FILES])
    def test_boundary_potential(path):
        check_boundary_potential(path)
except NameError:
    pass


if __name__ == "__main__":
    for path in EQ_FILES:
        check_boundary_potential(path)
