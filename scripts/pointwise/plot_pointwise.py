"""Figures of the pointwise-bound study (``al_run.py``, ``check_states.py``, ``hessian.py``,
``poincare_state.py``). Each subcommand writes figures/pointwise/<name>.png.

    front    label=run.json ... [--check check.json]   Pareto front |J - J_fit| against the
             largest force, with the multiplier slope check and |p - p0|
    tr       label=run.json ...                       trust-region history: radius, step,
             ratio, Gauss-Newton share, mu, per step (alpha changes marked)
    ladder   check.npz check.json                      convergence with the annulus h: the
             states' differences between successive resolutions, their force on the
             reference annulus, and the J, p profiles
    maps     tables.npz label=state.npz ...            the plane zeta = 0: multiplier,
             |F| / eps, |J - J_fit| / rms, p - p0
    poincare poincare.npz                              sections in DESC (theta, rho), iota and
             the spread of p along each line
    hessian  label=hess.npz ...                        eigenvalue spectra
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(ROOT, "figures", "pointwise")


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, name + ".png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    print("saved", p)


def _runs(specs):
    """label=run.json[+more.json...]: the alphas of continuations merged (sorted by alpha,
    descending; a feasible state replaces an infeasible one at the same alpha)."""
    out = {}
    for s in specs:
        lab, ps = s.split("=", 1)
        runs = [json.load(open(p)) for p in ps.split("+") if os.path.exists(p)]
        r = dict(runs[0])
        by = {}
        for q in runs:
            for e in q["alphas"]:
                al = e["metrics"]["alpha"]
                if al not in by or (e["metrics"]["feasible"] and not by[al]["metrics"]["feasible"]):
                    by[al] = e
        r["alphas"] = [by[k] for k in sorted(by, reverse=True)]
        out[lab] = r
    return out


def front(a):
    runs = _runs(a.runs)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for i, (lab, r) in enumerate(runs.items()):
        ms = [e["metrics"] for e in r["alphas"] if e["metrics"]["feasible"]]
        x = [m["max|F|"] for m in ms]
        col = f"C{i}"
        yJ = np.array([m["|J-J0|"] for m in ms])
        floor = 1e-5
        ax[0].loglog(x, np.maximum(yJ, floor), "o-", color=col, label=lab, ms=4)
        po = yJ < 1e-10
        if po.any():  # met by the pressure alone: |J - J_fit| at round-off
            ax[0].loglog(np.array(x)[po], np.full(po.sum(), floor), "v", color=col, ms=9,
                         mfc="none", label=f"{lab}: p only (J unchanged)")
        ax[1].loglog(x, [m["p-p0"] for m in ms], "o-", color=col, label=lab, ms=4)
        ax[1].loglog(x, [m["J-J0 along B"] for m in ms], "^:", color=col, ms=3)
        # slope check: dPhi/deps by differences of neighbouring alphas vs the multipliers'
        eps = np.array([m["eps"] for m in ms])
        Phi = np.array([m["Phi"] for m in ms])
        if len(eps) > 1:
            em = np.sqrt(eps[1:] * eps[:-1])
            fd = np.diff(Phi) / np.diff(eps)
            ax[2].loglog(em, -fd, "s", color=col, mfc="none", label=f"{lab}: differences")
        ax[2].loglog(eps, [-m["slope dPhi/deps"] for m in ms], "-", color=col,
                     label=f"{lab}: -(2/eps) sum w lam (1+c)")
    r0 = next(iter(runs.values()))
    for k, ls in (("max|F|", "--"),):
        ax[0].axvline(r0["start"][k], color="0.5", ls=ls, lw=1)
    ax[0].axvline(r0["F_ref"], color="C3", ls=":", lw=1, label="F_ref (DESC p99)")
    ax[0].set_xlabel("max |F| / norm (= eps)")
    ax[0].set_ylabel("rms |J - J_fit| / rms J_fit (floor 1e-5)")
    ax[0].set_title("Pareto front (dashed: start)")
    ax[1].set_xlabel("max |F| / norm")
    ax[1].set_ylabel("rms(p - p0) / rms p0 (o);  J - J_fit along B (^)")
    ax[1].set_title("pressure and parallel current")
    ax[2].set_xlabel("eps")
    ax[2].set_ylabel("-dPhi*/deps")
    ax[2].set_title("front slope: differences vs multipliers")
    ax[2].set_ylim(1e-7, 1e3)
    for x in ax:
        x.grid(alpha=0.3, which="both")
        x.invert_xaxis() if x is not ax[2] else None
    ax[0].legend(fontsize=8)
    ax[2].legend(fontsize=7)
    if a.check:
        ck = json.load(open(a.check))
        for i, (lab, d) in enumerate(ck["runs"].items()):
            ms = sorted(d.values(), key=lambda m: -m["eps"])
            ax[0].loglog([m["max|F|"] for m in ms], [max(m["|J-J0|"], 1e-5) for m in ms], "x--",
                         color=f"C{i + len(runs)}", label=f"{lab} (force on the reference annulus)")
        ax[0].legend(fontsize=7)
    _save(fig, a.name)


def tr(a):
    runs = _runs(a.runs)
    fig, ax = plt.subplots(5, 1, figsize=(13, 14), sharex=False)
    for i, (lab, r) in enumerate(runs.items()):
        k0 = 0
        rows = []
        bounds = []
        for e in r["alphas"]:
            for o in e["outer"]:
                for h in o["inner"]:
                    rows.append(h)
                bounds.append((len(rows), o["outer"] == 0, e["metrics"]["alpha"]))
        k = np.arange(len(rows))
        g = lambda key: np.array([h.get(key, np.nan) if h.get(key) is not None else np.nan  # noqa: E731
                                  for h in rows], float)
        col = f"C{i}"
        ax[0].semilogy(k, g("Delta"), "-", color=col, label=f"{lab}: radius Delta")
        ax[0].semilogy(k, g("|y|"), ".", color=col, ms=2)
        ax[1].plot(k, np.clip(g("ratio"), -1, 2), ".", color=col, ms=3, label=lab)
        ax[2].semilogy(k, g("GN share"), ".", color=col, ms=3, label=lab)
        ax[3].semilogy(k, g("mu"), ".", color=col, ms=3, label=lab)
        ax[4].semilogy(k, g("f"), "-", color=col, lw=0.8, label=lab)
        for b, first, al in bounds:
            if first:
                for x in ax:
                    x.axvline(b, color=col, lw=0.3, alpha=0.4)
        k0 += len(rows)
    ax[0].set_ylabel("Delta (line), |y| (dots)")
    ax[1].set_ylabel("actual / predicted")
    ax[2].set_ylabel("Gauss-Newton share")
    ax[3].set_ylabel("mu (0: undamped GN step)")
    ax[4].set_ylabel("AL objective f")
    ax[4].set_xlabel("trust-region step (all alphas, outer iterations; lines: AL updates)")
    for x in ax:
        x.grid(alpha=0.3)
        x.legend(fontsize=7)
    _save(fig, a.name)


def ladder(a):
    P = np.load(a.npz)
    ck = json.load(open(a.json))
    labels = a.labels or list(ck["runs"])  # in order of increasing resolution
    alphas = sorted(set.intersection(*[set(ck["runs"][l]) for l in labels]), key=lambda s: -float(s))
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    for lab in labels:
        ms = [ck["runs"][lab][s] for s in alphas]
        ax[0, 0].loglog([float(s) for s in alphas], [m["max|F|"] for m in ms], "o-", label=f"{lab} on ref")
        ax[0, 0].loglog([float(s) for s in alphas], [m["own max|F|"] for m in ms], "x:", label=f"{lab} own")
        ax[0, 1].loglog([float(s) for s in alphas], [m["rms|F|"] for m in ms], "o-", label=lab)
    w = P["weights"] if "weights" in P.files else None
    rmsw = lambda v: np.sqrt(np.sum(w * np.sum(np.reshape(v, (len(w), -1)) ** 2, 1)))  # noqa: E731
    steps = [f"{labels[i]}->{labels[i + 1]}" for i in range(len(labels) - 1)]
    for j, s in enumerate([q for q in alphas if float(q) in (2.0, 1.0, 0.5, 0.3)]):
        col = f"C{j}"
        if w is not None and f"{labels[0]}|{s}|Jpt" in P.files:
            dJ = [rmsw(P[f"{labels[i + 1]}|{s}|Jpt"] - P[f"{labels[i]}|{s}|Jpt"])
                  / rmsw(P[f"{labels[-1]}|{s}|Jpt"]) for i in range(len(labels) - 1)]
            dp = [rmsw(P[f"{labels[i + 1]}|{s}|ppt"] - P[f"{labels[i]}|{s}|ppt"])
                  / rmsw(P[f"{labels[-1]}|{s}|ppt"]) for i in range(len(labels) - 1)]
            ax[0, 2].semilogy(steps, dJ, "o-", color=col, label=f"alpha {s}: J at the points")
            ax[0, 2].semilogy(steps, dp, "s--", color=col, label=f"alpha {s}: p at the points")
        xs = [P[f"{l}|{s}|x"] for l in labels]
        d = [np.linalg.norm(xs[i + 1] - xs[i]) / np.linalg.norm(xs[-1]) for i in range(len(xs) - 1)]
        ax[0, 2].semilogy(steps, d, "x:", color=col, label=f"alpha {s}: coefficients x")
    s = alphas[min(len(alphas) - 1, len(alphas) // 2)]
    for lab in labels:
        rho = P[f"{lab}|{s}|rho"]
        ax[1, 0].plot(rho, P[f"{lab}|{s}|J.b"], label=lab)
        ax[1, 1].plot(rho, P[f"{lab}|{s}|p"], label=lab)
        ax[1, 2].semilogy(rho, P[f"{lab}|{s}|F| max"] if f"{lab}|{s}|F| max" in P else P[f"{lab}|{s}||F| max"],
                          label=lab)
    ax[1, 0].plot(P["start|rho"], P["start|J.b"], "k:", label="start")
    ax[1, 1].plot(P["start|rho"], P["start|p"], "k:", label="start")
    ax[0, 0].set_title("max |F| / norm vs alpha (o: on the reference annulus)")
    ax[0, 1].set_title("rms |F| / norm on the reference annulus")
    ax[0, 2].set_title("rms change between successive resolutions / rms")
    ax[1, 0].set_title(f"<J . b>(rho), alpha {s}")
    ax[1, 1].set_title(f"<p>(rho), alpha {s}")
    ax[1, 2].set_title(f"max |F| / norm on each surface (reference annulus), alpha {s}")
    for x in ax.ravel():
        x.grid(alpha=0.3, which="both")
        x.legend(fontsize=7)
    _save(fig, a.name)


def maps(a):
    T = np.load(a.tables, mmap_mode="r")
    shape = tuple(int(s) for s in T["tgrid_shape"])
    close = lambda v: np.concatenate([v, v[:, :1]], 1)  # noqa: E731  (theta periodic)
    R, Z = (close(np.asarray(T[k]).reshape(shape)[:, :, 0]) for k in ("R", "Z"))
    specs = [s.split("=", 1) for s in a.states]
    fig, ax = plt.subplots(len(specs), 4, figsize=(16, 4.2 * len(specs)), squeeze=False)
    for i, (lab, p) in enumerate(specs):
        st = np.load(p)
        eps = float(st["eps"])
        q = {"multiplier lam": np.asarray(st["lam"]), "|F| / eps": np.asarray(st["Fn"]) / eps,
             "|J - J_fit| / rms": np.asarray(st["dJ"]), "(p - p0) / rms p0": np.asarray(st["dp"])}
        for j, (k, v) in enumerate(q.items()):
            v = close(v.reshape(shape)[:, :, 0])
            if j == 0:  # multipliers: log scale
                from matplotlib.colors import LogNorm
                top = max(v.max(), 1e-300)
                c = ax[i, j].pcolormesh(R, Z, np.maximum(v, top * 1e-6), shading="gouraud",
                                        norm=LogNorm(top * 1e-6, top), cmap="viridis")
                fig.colorbar(c, ax=ax[i, j], shrink=0.8)
                ax[i, j].set_aspect("equal")
                ax[i, j].set_title(f"{lab}: multiplier lam (log)", fontsize=9)
                continue
            kw = {"cmap": "RdBu_r", "vmin": -abs(v).max(), "vmax": abs(v).max()} if k.startswith("(p") \
                else {"cmap": "magma" if j else "viridis"}
            if k == "|F| / eps":
                kw = {"cmap": "magma", "vmin": 0, "vmax": 1.0}
            c = ax[i, j].pcolormesh(R, Z, v, shading="gouraud", **kw)
            fig.colorbar(c, ax=ax[i, j], shrink=0.8)
            ax[i, j].set_aspect("equal")
            ax[i, j].set_title(f"{lab}: {k}", fontsize=9)
    _save(fig, a.name)


def poincare(a):
    P = np.load(a.npz)
    names = [k[:-5] for k in P.files if k.endswith("_iota")]
    fig, ax = plt.subplots(2, len(names), figsize=(5.5 * len(names), 9), squeeze=False)
    fig2, ax2 = plt.subplots(1, 2, figsize=(13, 4.5))
    for j, n in enumerate(names):
        rho, th, pp = P[f"{n}_rho"], P[f"{n}_theta"], P[f"{n}_p"]
        ax[0, j].scatter(np.mod(th, 2 * np.pi).ravel() / np.pi, rho.ravel(), s=0.05, c="k",
                         rasterized=True)
        ax[0, j].set_ylim(0.3, 1.0)
        ax[0, j].set_title(f"{n}: phi = 0 in DESC (theta, rho)")
        ax[0, j].set_xlabel("theta / pi")
        s = ax[1, j].scatter(P[f"{n}_R"].ravel(), P[f"{n}_Z"].ravel(), c=pp.ravel(), s=0.1,
                             cmap="viridis", rasterized=True)
        ax[1, j].set_aspect("equal")
        fig.colorbar(s, ax=ax[1, j], shrink=0.7)
        ax[1, j].set_title(f"{n}: p at the crossings")
        rl = np.nanmean(rho, 0)
        pmax = np.nanmax(pp)
        ax2[0].plot(rl, np.abs(P[f"{n}_iota"]), ".", ms=4, label=n)
        ax2[1].semilogy(rl, np.nanstd(pp, 0) / pmax, ".", ms=4, label=n)
    ax2[0].set_xlabel("<rho> of the line")
    ax2[0].set_ylabel("|iota|")
    ax2[1].set_xlabel("<rho> of the line")
    ax2[1].set_ylabel("std of p along the line / max p")
    for x in ax2:
        x.grid(alpha=0.3)
        x.legend(fontsize=8)
    _save(fig, a.name + "_sections")
    _save(fig2, a.name + "_iota_p")


def hessian(a):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for i, s in enumerate(a.hess):
        lab, p = s.split("=", 1)
        H = np.load(p)
        for j, k in enumerate(("H_AL", "GN_AL", "H_Lag_reduced")):
            if k not in H.files:
                continue
            ev = H[k]
            top = np.abs(ev).max()
            pos, neg = ev[ev > 0] / top, -ev[ev < -1e-10 * top] / top  # negatives beyond round-off
            ax[j].semilogy(np.arange(len(pos)), np.sort(pos)[::-1], "-", color=f"C{i}", label=f"{lab} (+)")
            if len(neg):
                ax[j].semilogy(np.arange(len(neg)), np.sort(neg)[::-1], ":", color=f"C{i}",
                               label=f"{lab} (-) x{len(neg)}")
            ax[j].set_title(k)
    for x in ax:
        x.set_xlabel("index (descending)")
        x.set_ylabel("|eigenvalue| / max  (dotted: negative, < -1e-10 max)")
        x.grid(alpha=0.3)
        x.legend(fontsize=7)
    _save(fig, a.name)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("front"); p.add_argument("runs", nargs="+"); p.add_argument("--check"); p.add_argument("--name", default="front")  # noqa: E702
    p = sub.add_parser("tr"); p.add_argument("runs", nargs="+"); p.add_argument("--name", default="tr")  # noqa: E702
    p = sub.add_parser("ladder"); p.add_argument("npz"); p.add_argument("json"); p.add_argument("--labels", nargs="+"); p.add_argument("--name", default="ladder")  # noqa: E702
    p = sub.add_parser("maps"); p.add_argument("tables"); p.add_argument("states", nargs="+"); p.add_argument("--name", default="maps")  # noqa: E702
    p = sub.add_parser("poincare"); p.add_argument("npz"); p.add_argument("--name", default="poincare")  # noqa: E702
    p = sub.add_parser("hessian"); p.add_argument("hess", nargs="+"); p.add_argument("--name", default="hessian")  # noqa: E702
    a = ap.parse_args()
    {"front": front, "tr": tr, "ladder": ladder, "maps": maps, "poincare": poincare,
     "hessian": hessian}[a.cmd](a)


if __name__ == "__main__":
    main()
