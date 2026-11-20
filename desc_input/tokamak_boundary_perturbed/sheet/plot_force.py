import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "diag/"
runs = [("ripple_L32N2_x_it3000", "L=M=32, N=2", "tab:blue", "-"), ("ripple_L32N4_d", "L=M=32, N=4", "tab:blue", ":"),
        ("ripple_L48N2_x_it3000", "L=M=48, N=2", "tab:orange", "-"),
        ("ripple_L64N2_x_it1300", "L=M=64, N=2", "tab:green", "-"), ("ripple_L64N5_x_it80", "L=M=64, N=5", "tab:green", ":")]
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
for t, lab, c, ls in runs:
    z = np.load(D + t + "_sheet.npz"); r, f, rs = z["rho"], z["F"] / z["Fn"], float(z["rho_s"])
    for a_ in ax:
        a_.semilogy(r, f, color=c, ls=ls, lw=1.3, label=lab)
for a_ in ax:
    a_.axvline(rs, color="k", ls=":", lw=0.8); a_.axvline(0.9131, color="gray", ls=":", lw=0.8)
    a_.set(xlabel=r"$\rho$", ylabel=r"$\langle|F|\rangle_s / \langle|\nabla B^2/2\mu_0|\rangle_s$")
ax[0].text(rs + 0.005, 2e-8, "q=2", fontsize=9); ax[0].text(0.9131 + 0.005, 2e-8, "q=3", fontsize=9, color="gray")
ax[0].set(title="force error by flux surface (iteration-converged states)", ylim=(1e-8, 3e-6))
ax[1].set(title="zoom on q=2", xlim=(rs - 0.1, rs + 0.1), ylim=(3e-8, 2e-6))
ax[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(D + "force_by_rho.png", dpi=130)
