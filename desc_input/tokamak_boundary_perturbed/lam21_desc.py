import sys, json, numpy as np
from desc.io import load
from desc.grid import LinearGrid
out = {}
rho = np.round(np.arange(0.30, 0.951, 0.01), 4)
for f in sys.argv[1:]:
    eq = load(f); eq = eq[-1] if hasattr(eq, "__len__") else eq
    amp, ph, l00 = [], [], []
    for r in rho:
        g = LinearGrid(rho=np.array([r]), M=48, N=12, NFP=eq.NFP, sym=False)
        d = eq.compute(["J", "B", "lambda", "lambda_t"], grid=g)
        th, ze = g.nodes[:, 1], g.nodes[:, 2]
        lam = np.sum(d["J"] * d["B"], 1) / np.sum(d["B"] ** 2, 1)
        w = 1 + d["lambda_t"]
        c = 2 * np.mean(lam * np.exp(-1j * (2 * (th + d["lambda"]) + ze)) * w)
        amp.append(float(abs(c))); ph.append(float(np.degrees(np.angle(c)))); l00.append(float(np.mean(lam * w)))
    out[f] = dict(amp=amp, phase=ph, l00=l00)
    a = np.array(amp)
    print(f.split('/')[-1], ' '.join(f"{x:.2f}:{y:.0f}/{p:.0f}" for x, y, p in zip(rho[::3], a[::3], np.array(ph)[::3])), flush=True)
json.dump(dict(rho=rho.tolist(), **out), open("/tmp/lam21_desc.json", "w"))
