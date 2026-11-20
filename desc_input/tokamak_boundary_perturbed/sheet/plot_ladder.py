import numpy as np, matplotlib, json
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "diag/"
def load(t):
    z = np.load(D + t + "_sheet.npz"); j = json.load(open(D + t + "_sheet.json"))
    return z["rho"], z["s21"], float(z["rho_s"]), j
def signed(r, s, rs):
    return np.real(s * np.exp(-1j * np.angle(s[np.argmin(abs(r - rs))])))
lad = [("ripple_L32N2_x_it3000", 32, "tab:blue"), ("ripple_L48N2_x_it3000", 48, "tab:orange"),
       ("ripple_L64N2_x_it1300", 64, "tab:green")]
fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
rows = []
for t, L, c in lad:
    r, s, rs, j = load(t); sg = signed(r, s, rs); x = r - rs
    ax[0, 0].plot(r, abs(s), color=c, lw=1.3, label=f"L=M={L}")
    ax[0, 1].plot(x, sg, color=c, lw=1.3, label=f"L=M={L}")
    m = abs(x) <= 0.08
    cum = np.concatenate([[0], np.cumsum(0.5 * (sg[m][1:] + sg[m][:-1]) * np.diff(x[m]))])
    ax[0, 2].plot(x[m], cum - np.interp(0, x[m], cum), color=c, lw=1.3, label=f"L=M={L}")
    ax[1, 0].plot(x * L, sg / L**2, color=c, lw=1.3, label=f"L=M={L}")
    rows.append((L, j))
ax[0, 0].axvline(rs, color="k", ls=":", lw=0.8)
ax[0, 0].set(title="converged $|\\sigma_{21}|(\\rho)$, N=2", xlabel=r"$\rho$", ylabel=r"$|\sigma_{21}|$ [A/(T m$^2$)]")
ax[0, 1].axvline(0, color="k", ls=":", lw=0.8); ax[0, 1].axhline(0, color="k", lw=0.5)
ax[0, 1].set(xlim=(-0.2, 0.2), title="signed, phase at q=2", xlabel=r"$\rho-\rho_s$",
             ylabel=r"Re$[\sigma_{21}e^{-i\phi(\rho_s)}]$")
ax[0, 2].axvline(0, color="k", ls=":", lw=0.8)
ax[0, 2].set(title=r"running integral $\int_{\rho_s}^{\rho}\sigma_{21}d\rho'$ (a sheet = a step)",
             xlabel=r"$\rho-\rho_s$")
ax[1, 0].set(xlim=(-10, 10), title=r"collapse test: $\sigma_{21}/L^2$ vs $L(\rho-\rho_s)$",
             xlabel=r"$L(\rho-\rho_s)$", ylabel=r"$\sigma_{21}/L^2$")
Ls = np.array([L for L, _ in rows])
pk = np.array([j["peak |sigma_21|"] for _, j in rows]); fw = np.array([j["FWHM"] for _, j in rows])
ax[1, 1].loglog(Ls, pk, "o-", label="peak"); ax[1, 1].loglog(Ls, pk[0] * (Ls / Ls[0]) ** 2, "k:", label=r"$\propto L^2$")
ax[1, 1].loglog(Ls, 1e4 * fw, "s-", label=r"FWHM $\times 10^4$"); ax[1, 1].loglog(Ls, 1e4 * fw[0] * Ls[0] / Ls, "k--", label=r"$\propto 1/L$")
ax[1, 1].set(title="scaling with L", xlabel="L = M"); ax[1, 1].legend(fontsize=8)
for k, mk in (("0.01", "o-"), ("0.02", "s-"), ("0.04", "^-"), ("0.08", "v-")):
    ax[1, 2].plot(Ls, [j["integral signed"][k] for _, j in rows], mk, label=rf"$|\rho-\rho_s|\leq{k}$")
r, s, rs, j5 = load("ripple_L64N5_x_it80")
ax[1, 2].plot([64], [j5["integral signed"]["0.02"]], "r*", ms=12, label="L=64, N=5 (±0.02)")
ax[1, 2].set(title=r"sheet strength $\int\sigma_{21}d\rho$ across q=2", xlabel="L = M"); ax[1, 2].legend(fontsize=8)
for a_ in (ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0]):
    a_.legend(fontsize=8)
fig.tight_layout(); fig.savefig(D + "sigma21_ladder.png", dpi=125)
for L, j in rows:
    print(L, j["peak |sigma_21|"], j["FWHM"], j["integral signed"])
