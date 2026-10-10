"""Preconditioners for the Gauss-Newton system A d = -g of ``spectrum.py``.

Loads A = J^T J, g and the mode labels (l, m, n) of u from a ``spectrum.py``
npz and runs preconditioned CG on the exact A with
  jacobi        diag(A) (the optimizer's choice)
  block (m, n)  all unknowns sharing a Fourier mode (every l of a and b, and chi)
  block |m|,|n| the same, merging the signs of m and n
  block n       all unknowns sharing n
  block m       all unknowns sharing m
  probed s      block |m|,|n| estimated as with J v only: one product per color
                (|m| mod s, |n| mod s) and slot (position in the block), each
                with one column of every block of that color, random signs
For each: the condition number of the preconditioned A (on its range) and the
model decrease after k iterations, relative to the exact minimum.

  local         blocks of the local columns (``local_blocks.py`` npz, second argument)

run: python scripts/precond_test.py results/spectrum16.npz [results/blocks16.npz]
"""

import sys

import numpy as np
import scipy.linalg as sl


def blocks(labels, key):
    keys = {}
    for i, lab in enumerate(labels):
        keys.setdefault(key(lab), []).append(i)
    return [np.array(v) for v in keys.values()]


def block_inverse(A, groups, floor=1e-10):
    """Apply the inverse of the block diagonal of A (eigen-decomposed blocks,
    eigenvalues floored at floor x the block's largest)."""
    facs = []
    for idx in groups:
        w, V = np.linalg.eigh(A[np.ix_(idx, idx)])
        w = np.maximum(w, floor * w[-1])
        facs.append((idx, V, w))

    def apply(r):
        z = np.zeros_like(r)
        for idx, V, w in facs:
            z[idx] = V @ ((V.T @ r[idx]) / w)
        return z

    def half(Mat):  # B^-1/2 A B^-1/2 for the condition number
        S = np.zeros_like(Mat)
        for idx, V, w in facs:
            S[np.ix_(idx, idx)] = V @ np.diag(w**-0.5) @ V.T
        return S @ Mat @ S

    return apply, half


def probed_blocks(Amul, labels, stride, seed=0):
    """Estimate the |m|,|n| diagonal blocks of A from products A v only."""
    groups = blocks(labels, lambda t: (abs(t[1]), abs(t[2])))
    keys = [(abs(labels[b[0], 1]), abs(labels[b[0], 2])) for b in groups]
    rng = np.random.default_rng(seed)
    est = [np.zeros((len(b), len(b))) for b in groups]
    colors = {}
    for i, (m, n) in enumerate(keys):
        colors.setdefault((m % stride, n % stride), []).append(i)
    n_prod = 0
    for members in colors.values():
        for j in range(max(len(groups[i]) for i in members)):
            v = np.zeros(len(labels))
            sg = {}
            for i in members:
                if j < len(groups[i]):
                    sg[i] = rng.choice([-1.0, 1.0])
                    v[groups[i][j]] = sg[i]
            y = Amul(v)
            n_prod += 1
            for i, si in sg.items():
                est[i][:, j] = si * y[groups[i]]
    est = [0.5 * (E + E.T) for E in est]
    return groups, est, n_prod


def block_inverse_from(groups, mats, floor=1e-10):
    facs = []
    for idx, Ab in zip(groups, mats):
        w, V = np.linalg.eigh(Ab)
        w = np.maximum(w, floor * max(w[-1], 1e-300))
        facs.append((idx, V, w))

    def apply(r):
        z = np.zeros_like(r)
        for idx, V, w in facs:
            z[idx] = V @ ((V.T @ r[idx]) / w)
        return z

    def half(Mat):
        S = np.zeros_like(Mat)
        for idx, V, w in facs:
            S[np.ix_(idx, idx)] = V @ np.diag(w**-0.5) @ V.T
        return S @ Mat @ S

    return apply, half


def pcg_model(A, g, Minv, ks):
    """Model decrease -2 g.d - d.A.d of PCG iterates at the iteration counts ks."""
    x = np.zeros_like(g)
    r = -g.copy()
    z = Minv(r)
    p = z.copy()
    rz = r @ z
    out = {}
    for k in range(1, max(ks) + 1):
        Ap = A @ p
        if p @ Ap <= 0:  # converged
            out.update({kk: out.get(k - 1, -2 * g @ x - x @ A @ x) for kk in ks if kk >= k})
            break
        alpha = rz / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        if k in ks:
            out[k] = -2 * g @ x - x @ A @ x
        z = Minv(r)
        rz_new = r @ z
        p = z + rz_new / rz * p
        rz = rz_new
    return out


def main(path):
    d = np.load(path)
    A, g, lab = d["A"], d["g"], d["labels"]
    n = len(g)
    w, V = np.linalg.eigh(A)
    keep = w > 1e-13 * w[-1]
    opt = float(((V.T @ g)[keep] ** 2 / w[keep]).sum())
    print(f"{path}: n = {n}, |r|^2 = {float(d['r2']):.3e}, optimal decrease {opt:.3e} "
          f"({opt / float(d['r2']):.3f} of |r|^2)")
    D = np.diag(A)
    pre = {"jacobi": ((lambda r: r / D), None)}
    for name, key in [("block (m, n)", lambda t: (t[1], t[2])),
                      ("block |m|,|n|", lambda t: (abs(t[1]), abs(t[2]))),
                      ("block n", lambda t: t[2]), ("block m", lambda t: t[1])]:
        groups = blocks(lab, key)
        if max(len(b) for b in groups) > 4000:
            continue
        pre[name] = block_inverse(A, groups)
        pre[name] = (pre[name][0], pre[name][1], len(groups), max(len(b) for b in groups))
    for stride in (2, 3, 4, 6):
        groups, est, n_prod = probed_blocks(lambda v: A @ v, lab, stride)
        ap, hf = block_inverse_from(groups, est)
        pre[f"probed s={stride} ({n_prod} Av)"] = (ap, hf, len(groups), max(len(b) for b in groups))
    if len(sys.argv) > 2:
        b = np.load(sys.argv[2])
        groups = [row[row < n] for row in b["idx"]]
        mats = [(Vg[: len(gr), : len(gr)] * wg[: len(gr)]) @ Vg[: len(gr), : len(gr)].T
                for gr, Vg, wg in zip(groups, b["V"], b["w"])]
        ap, hf = block_inverse_from(groups, mats)
        pre["local blocks"] = (ap, hf, len(groups), max(len(gr) for gr in groups))
    ks = [10, 30, 100, 300, 1000, 3000]
    print(f"{'preconditioner':26s} {'blocks':>7s} {'max':>5s} {'cond':>9s}  "
          + "  ".join(f"k={k:<5d}" for k in ks))
    for name, p in pre.items():
        Minv = p[0]
        if name == "jacobi":
            S = A / np.sqrt(np.outer(D, D))
            nb, mx = n, 1
        else:
            S = p[1](A)
            nb, mx = p[2], p[3]
        e = sl.eigvalsh(S)
        cond = e[-1] / e[e > 1e-13 * e[-1]][0]
        m = pcg_model(A, g, Minv, set(ks))
        print(f"{name:26s} {nb:7d} {mx:5d} {cond:9.2e}  "
              + "  ".join(f"{m[k] / opt:7.4f}" for k in ks), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
