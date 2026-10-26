"""Report of the speed-elim production A/B (docs/pressure_poisson_runbook.md): one markdown file from the scans'
results/isl/<name>_eps<eps>.json, the conditioning jsons and the Slurm logs.

usage (repo root):
  python scripts/lm/elim_report.py --runs abB=abN2B1000 abD=abN2D1000 abE=abN2E1000 --eps 1000 \
      --ref-state a120Cn2d1000 --cond results/isl/cond_a120Cn2d1000.json [results/isl/cond_a120Cn2d1000_p3.json] \
      --logs abB=logs/nest-111.log abD=logs/nest-112.log abE=logs/nest-113.log cond=logs/nest-114.log \
      --out results/isl/elim_report.md

All f values are divided by f_ref, the joint f of the common start state (the reference run's iteration 0, else the
last f of --ref-state), because an elim run's own iteration 0 is recorded after its start projection (lower f).
"""
import argparse
import datetime
import gzip
import json
import os
import re
import socket
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ISL = os.path.join(ROOT, "results", "isl")
THRESH = {"|c_p|/|r|": 1e-8, "f* check": 1e-8, "p min/max": -1e-2}


def _git():
    try:
        h = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no"], text=True)
        return h + (" (uncommitted changes)" if dirty.strip() else "")
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e})"


def _read_log(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", errors="replace") as fh:
        return fh.read()


def log_facts(text):
    """GPU, table cache use, errors, stops and warnings from a Slurm log."""
    out = {}
    m = re.search(r"NVIDIA [^,\n]+", text)
    out["gpu"] = m.group(0).strip() if m else "?"
    m = re.search(r"JAX devices: (.*)", text)
    out["jax devices"] = m.group(1).count("CudaDevice") if m else "?"
    out["tables from cache"] = re.findall(r"table (\w+): \([^)]*\) from the cache", text)
    out["tables built"] = re.findall(r"table (\w+): \([^)]*\), \d+ s", text)
    out["box solves estimated"] = re.findall(r"([\d.]+ min estimated for \d+ box solves)", text)
    out["traceback"] = "Traceback" in text
    out["errors"] = [ln.strip() for ln in text.splitlines()
                     if re.search(r"(Error|error:|Killed|CANCELLED|DUE TO TIME LIMIT|out of memory|RESOURCE_EXHAUSTED)", ln)][-5:]
    out["stops"] = re.findall(r"it +\d+: no decrease \(\w+\); stopping", text)
    out["elim warnings"] = re.findall(r"elim WARNING: .*", text)
    out["Rpp"] = re.findall(r"elim: R_pp \(\d+\) built in [\d.]+ s", text)
    out["unknowns"] = re.findall(r"harmonic model: N = \d+, collocation \([^)]*\), ([\d,]+) unknowns", text)
    return out


def run_facts(name, eps):
    p = os.path.join(ISL, f"{name}_eps{eps:g}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    h = d["history"]
    sp = d["args"].get("speed") or ""
    return {"name": name, "path": os.path.relpath(p, ROOT), "hist": h, "done": d.get("done"), "speed": sp,
            "p_factors": d["args"].get("p_factors"), "elim": "elim" in sp.split(",")}


def at(h, k, key="f"):
    for e in h:
        if e["it"] == k:
            return e.get(key)
    return None


def fmt(v, f=".3e"):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="label=name, e.g. abB=abN2B1000 (first = the reference)")
    ap.add_argument("--eps", type=float, default=1000.0)
    ap.add_argument("--ref-state", default="a120Cn2d1000")
    ap.add_argument("--cond", nargs="*", default=[])
    ap.add_argument("--logs", nargs="*", default=[], help="label=path of the Slurm logs (label as in --runs, or cond)")
    ap.add_argument("--out", default=os.path.join(ISL, "elim_report.md"))
    a = ap.parse_args()
    runs = [tuple(s.split("=", 1)) for s in a.runs]
    logs = dict(tuple(s.split("=", 1)) for s in a.logs)
    facts = {lab: run_facts(nm, a.eps) for lab, nm in runs}
    lf = {lab: log_facts(_read_log(p)) for lab, p in logs.items() if os.path.exists(p)}
    missing_logs = [p for p in logs.values() if not os.path.exists(p)]
    ref_lab = runs[0][0]
    f_ref, ref_src, pre_flags = None, "", []
    p = os.path.join(ISL, f"{a.ref_state}_eps{a.eps:g}.json")
    if os.path.exists(p):
        f_ref, ref_src = json.load(open(p))["history"][-1]["f"], f"last f of {a.ref_state}"
    if facts.get(ref_lab) and not facts[ref_lab]["elim"]:
        f0 = facts[ref_lab]["hist"][0]["f"]
        if f_ref is None:
            f_ref, ref_src = f0, f"{ref_lab} iteration 0"
        elif abs(f0 - f_ref) > 1e-6 * f_ref:
            pre_flags.append(f"{ref_lab} iteration 0 f {f0:.6e} != last f of {a.ref_state} {f_ref:.6e}: different start "
                             "state or objective flags (check --start-from and the COMMON flags)")
    if f_ref is None:
        pre_flags.append(f"no f_ref: neither results/isl/{a.ref_state}_eps{a.eps:g}.json nor a joint reference run")
    L = []
    w = L.append
    w("# speed elim production A/B: report\n")
    w(f"- generated {datetime.datetime.now().isoformat(timespec='minutes')} on {socket.gethostname()}; code {_git()}")
    w(f"- eps {a.eps:g} A; f_ref = {fmt(f_ref, '.6e')} ({ref_src}); all f below are f / f_ref\n")
    flags = list(pre_flags)
    w("## 1. Runs and health\n")
    w("| label | name | speed | p-factors | its | wall (min) | median s/it | max abs(c_p)/abs(r) | max f* chk | p min/max (end) | GPU (log) | tables |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lab, nm in runs:
        r, g = facts.get(lab), lf.get(lab, {})
        if r is None:
            w(f"| {lab} | {nm} | - | - | MISSING results/isl/{nm}_eps{a.eps:g}.json | | | | | | | |")
            flags.append(f"{lab}: no result json (job failed or still running?)")
            continue
        h = r["hist"][1:]
        ts = [e["time"] for e in h]
        cp = max([e.get("|c_p|/|r|", 0.0) for e in h] or [0.0]) if r["elim"] else None
        fc = max([e.get("f* check", 0.0) for e in h] or [0.0]) if r["elim"] else None
        pm = h[-1].get("p min/max") if (h and r["elim"]) else None
        tab = ("cache " + ",".join(g.get("tables from cache", [])) if g.get("tables from cache") else "") + \
              (" built " + ",".join(g.get("tables built", [])) if g.get("tables built") else "")
        w(f"| {lab} | {nm} | {r['speed']} | {r['p_factors'] or 'default'} | {len(h)} | {sum(ts) / 60:.1f} | "
          f"{fmt(float(np.median(ts)) if ts else None, '.1f')} | {fmt(cp, '.1e')} | {fmt(fc, '.1e')} | {fmt(pm, '.1e')} | "
          f"{g.get('gpu', '-')} x{g.get('jax devices', '-')} | {tab or '-'} |")
        if r["elim"]:
            if cp is not None and cp > THRESH["|c_p|/|r|"]:
                flags.append(f"{lab}: max |c_p|/|r| = {cp:.1e} > {THRESH['|c_p|/|r|']:g} (p not optimal; report, do not rerun)")
            if fc is not None and fc > THRESH["f* check"]:
                flags.append(f"{lab}: max f* chk = {fc:.1e} > {THRESH['f* check']:g} (trial f vs direct evaluation)")
            if pm is not None and pm < THRESH["p min/max"]:
                flags.append(f"{lab}: p min/max = {pm:.1e} < {THRESH['p min/max']:g} (large negative pressure)")
        if len(h) < 30:
            flags.append(f"{lab}: only {len(h)} iterations recorded")
        for k in ("traceback",):
            if g.get(k):
                flags.append(f"{lab}: a Python traceback in the log")
        for e in g.get("errors", []):
            flags.append(f"{lab}: log line: {e[:200]}")
        for e in g.get("stops", []):
            flags.append(f"{lab}: {e}")
        for e in g.get("elim warnings", []):
            flags.append(f"{lab}: {e}")
        if g.get("box solves estimated"):
            flags.append(f"{lab}: tables were BUILT, not loaded from the cache ({'; '.join(g['box solves estimated'])}): NEST_CACHE?")
    for p in missing_logs:
        flags.append(f"log not found: {p}")
    w("\n## 2. Convergence (f / f_ref)\n")
    its = [0, 1, 5, 10, 15, 20, 25, 30]
    w("| label | " + " | ".join(f"it {k}" for k in its) + " | <abs(F)>/norm end | GN share end | |")
    w("|---|" + "---|" * len(its) + "---|---|---|")
    for lab, nm in runs:
        r = facts.get(lab)
        if r is None or f_ref is None:
            continue
        h = r["hist"]
        w(f"| {lab} | " + " | ".join(fmt(at(h, k) / f_ref if at(h, k) is not None else None, ".4f") for k in its)
          + f" | {fmt(h[-1].get('<|F|>/norm'))} | {fmt(h[-1].get('GN share'), '.2e')} | |")
    ref = facts.get(ref_lab)
    if ref is not None and f_ref is not None:
        w(f"\nAgainst the reference {ref_lab}:\n")
        w("| label | f / f_ref(" + ref_lab + ") at it 10 | at it 20 | at it 30 | first it with f <= " + ref_lab
          + "'s final f | its cumulative wall time (min) | " + ref_lab + " wall time (min) |")
        w("|---|---|---|---|---|---|---|")
        fB = ref["hist"][-1]["f"]
        tB = sum(e["time"] for e in ref["hist"][1:]) / 60
        for lab, nm in runs[1:]:
            r = facts.get(lab)
            if r is None:
                continue
            h = r["hist"]
            rat = [(at(h, k) / at(ref["hist"], k)) if (at(h, k) and at(ref["hist"], k)) else None for k in (10, 20, 30)]
            k_hit = next((e["it"] for e in h if e["f"] <= fB), None)
            t_hit = sum(e["time"] for e in h[1:] if e["it"] <= k_hit) / 60 if k_hit else None
            w(f"| {lab} | {fmt(rat[0], '.3f')} | {fmt(rat[1], '.3f')} | {fmt(rat[2], '.3f')} | {k_hit if k_hit else 'not reached'} | "
              f"{fmt(t_hit, '.1f')} | {tB:.1f} |")
    w("\n## 3. Conditioning (pressure_elim_conditioning_isl.py)\n")
    if a.cond:
        w("| file | n_u | n_p | f | kappa Jacobi joint | reduced | kappa phys joint | reduced | u-only | GN share joint / reduced | time (min) |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        for p in a.cond:
            if not os.path.exists(p):
                w(f"| {p} | MISSING | | | | | | | | | |")
                flags.append(f"cond: {p} missing")
                continue
            d = json.load(open(p))
            k = d["kappa"]
            w(f"| {os.path.basename(p)} | {k['n_u']} | {k['n_p']} | {fmt(k['f'])} | {fmt(k['joint_jacobi'], '.2e')} | "
              f"{fmt(k['reduced_jacobi'], '.2e')} | {fmt(k['joint_phys'], '.2e')} | {fmt(k['reduced_phys'], '.2e')} | "
              f"{fmt(k['u_only_phys'], '.2e')} | {fmt(k['gn_share_joint'], '.2e')} / {fmt(k['gn_share_reduced'], '.2e')} | "
              f"{d.get('time', 0) / 60:.1f} |")
    else:
        w("(no conditioning json given)")
    w("\n## 4. Flags (automatic)\n")
    w("\n".join(f"- {f}" for f in flags) if flags else "- none")
    w("\n## 5. Files for the return package\n")
    for lab, nm in runs:
        for ext in (".json", ".npz"):
            p = os.path.join(ISL, f"{nm}_eps{a.eps:g}{ext}")
            w(f"- {os.path.relpath(p, ROOT)}: " + (f"{os.path.getsize(p) / 1e6:.2f} MB" if os.path.exists(p) else "MISSING"))
    for p in list(a.cond) + list(logs.values()):
        w(f"- {p}: " + (f"{os.path.getsize(p) / 1e6:.2f} MB" if os.path.exists(p) else "MISSING"))
    w("\n## 6. Notes from the operator\n")
    notes = os.path.join(ISL, "elim_operator_notes.md")
    w(open(notes).read().strip() if os.path.exists(notes) else "(none: results/isl/elim_operator_notes.md not written)")
    open(a.out, "w").write("\n".join(L) + "\n")
    print(f"-> {a.out}\n" + ("FLAGS:\n" + "\n".join("  " + f for f in flags) if flags else "no flags"))


if __name__ == "__main__":
    main()
