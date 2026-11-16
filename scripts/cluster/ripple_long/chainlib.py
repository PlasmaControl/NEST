"""Bookkeeping for chained long runs of scripts/meanforce/mf_fixed.py (numpy + stdlib only).

A *run* lives in RUNS/<RUN>/ and is a sequence of *segments* seg01, seg02, ... (one scheduler job
each). Every segment is a fresh mf_fixed.py process that starts from the newest checkpoint of the
previous segments. This module picks that checkpoint, summarizes finished segments, decides
whether the chain should continue, prints a live health digest and builds the final report.

usage (from segment.sh / status.sh / collect.sh; see docs/ripple_long_runs_runbook.md):
  python chainlib.py newest-state RUN_DIR [PHASE]       path of the newest checkpoint (empty if none)
  python chainlib.py strip IN.npz OUT.npz               copy with only u and c_p (fresh multiplier, mu = --rho0)
  python chainlib.py remaining-alphas RUN_DIR a1 a2 ..  the alphas without a finished <tag>_a<alpha>.npz
  python chainlib.py summarize SEG_DIR                  write SEG_DIR/summary.json, append RUN_DIR/chain.tsv
  python chainlib.py decide RUN_DIR RATE MAXSEG         'continue' or 'finished: <reason>'
  python chainlib.py status RUN_DIR [N]                 live digest of the current segment
  python chainlib.py report RUNS RUN [RUN ...]          RUNS/report/{report.md, steps_<RUN>.csv, diag_<RUN>.csv}
"""
import glob
import json
import os
import re
import sys
import time

import numpy as np

STATE_RE = re.compile(r"_(budget|a[0-9.]+_latest|a[0-9.]+|a[0-9.]+_o[0-9]+)\.npz$")
STEP_RE = re.compile(r"^\s+it\s+(\d+) f (\S+) -> (\S+) ratio (\S+) \|y\| (\S+) Delta (\S+) \|g\| (\S+) GN (\S+)( hit)? "
                     r"M/eps (\S+) .* (\S+)s$")


def segs(run_dir):
    return sorted(d for d in glob.glob(os.path.join(run_dir, "seg[0-9][0-9]")) if os.path.isdir(d))


def seg_phase(seg):
    p = os.path.join(seg, "PHASE")
    return open(p).read().strip() if os.path.exists(p) else ""


def newest_state(run_dir, phase=None):
    cands = []
    for s in segs(run_dir):
        if phase and seg_phase(s) != phase:
            continue
        cands += [f for f in glob.glob(os.path.join(s, "*.npz")) if STATE_RE.search(f)]
    good = []
    for f in cands:
        try:  # a checkpoint must load and hold u (writes are atomic, but check anyway)
            with np.load(f) as z:
                if z["u"].size > 0 and np.all(np.isfinite(z["u"])):
                    good.append(f)
        except Exception:
            pass
    return max(good, key=os.path.getmtime) if good else ""


def strip(src, dst):
    with np.load(src) as z:
        np.savez(dst, u=z["u"], c_p=z["c_p"])


def remaining_alphas(run_dir, alphas):
    done = set()
    for s in segs(run_dir):
        for f in glob.glob(os.path.join(s, "*_a*.npz")):
            m = re.search(r"_a([0-9.]+)\.npz$", f)
            if m:
                done.add(float(m.group(1)))
    return [a for a in alphas if float(a) not in done]


def _hists(run):
    """All inner-step records of a run JSON, in order, with their alpha."""
    out = []
    for e in run.get("alphas", []):
        hs = [o.get("inner", []) for o in e.get("outer", [])] + [e.get("inner (current outer)", [])]
        for h in hs:
            out += [(e["alpha"], x) for x in h]
    return out


def seg_json(seg):
    fs = [f for f in glob.glob(os.path.join(seg, "*.json")) if not f.endswith("summary.json")]
    if not fs:
        return None
    try:
        return json.load(open(max(fs, key=os.path.getmtime)))
    except Exception:
        return None


def summarize(seg):
    run_dir = os.path.dirname(seg.rstrip("/"))
    run = seg_json(seg) or {}
    H = _hists(run)
    acc = [(a, h) for a, h in H if h.get("accepted")]
    meps = [1 + h["c"][0] for a, h in acc]
    tsolve = float(sum(h.get("time", 0.0) for a, h in H))
    diag = (run.get("diag log") or [None])[-1]
    log_ = os.path.join(seg, "run.log")
    err = ""
    if os.path.exists(log_):
        txt = open(log_, errors="replace").read()
        for pat in (r"Traceback", r"RESOURCE_EXHAUSTED", r"out of memory", r"\bnan\b", r"\bKilled\b"):
            if re.search(pat, txt):
                err += pat.replace("\\b", "") + ";"
    s = {"segment": os.path.basename(seg), "phase": seg_phase(seg), "stage": run.get("stage", "no JSON"),
         "alphas": [e["alpha"] for e in run.get("alphas", [])], "steps": len(H), "accepted": len(acc),
         "rejected": len(H) - len(acc), "M/eps first": meps[0] if meps else None,
         "M/eps last": meps[-1] if meps else None, "solve s": tsolve, "total s": run.get("total [s]"),
         "setup s": float(sum(v for k, v in run.get("setup", {}).items() if isinstance(v, (int, float)))),
         "last diag": diag, "errors": err, "finished at": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(s, open(os.path.join(seg, "summary.json"), "w"), indent=1)
    tsv = os.path.join(run_dir, "chain.tsv")
    new = not os.path.exists(tsv)
    with open(tsv, "a") as fh:
        if new:
            fh.write("segment\tphase\tstage\talpha\tsteps\taccepted\trejected\tM/eps first\tM/eps last\tsolve_h\t"
                     "M/M_desc\tb/vac\tband p16/DESC\terrors\n")
        d = diag or {}
        g = lambda v: f"{v:.6g}" if isinstance(v, float) else v  # noqa: E731
        fh.write("\t".join(str(g(x)) for x in [
            s["segment"], s["phase"], s["stage"], s["alphas"][-1] if s["alphas"] else "", s["steps"], s["accepted"],
            s["rejected"], s["M/eps first"], s["M/eps last"], round(tsolve / 3600, 3), d.get("M/M_desc", ""),
            d.get("b/vac", ""), d.get("band p16/DESC", ""), err]) + "\n")
    print(json.dumps(s, default=str))
    return s


def decide(run_dir, rate, maxseg):
    """Stop when two consecutive floor-phase segments each lowered M/eps by less than RATE (relative, per hour of
    solve time), or after MAXSEG segments, or after an error."""
    S = []
    for s in segs(run_dir):
        p = os.path.join(s, "summary.json")
        if os.path.exists(p):
            S.append(json.load(open(p)))
    if not S:
        return "continue"
    if S[-1].get("errors"):
        return f"finished: error in {S[-1]['segment']} ({S[-1]['errors']})"
    if len(S) >= maxseg:
        return f"finished: reached {maxseg} segments"
    fl = [s for s in S if s.get("phase", "").startswith("floor")]

    def slow(s):
        a, b, h = s.get("M/eps first"), s.get("M/eps last"), s.get("solve s", 0) / 3600
        return a is not None and b is not None and h > 0.5 and (a - b) / a / h < rate

    if len(fl) >= 2 and slow(fl[-1]) and slow(fl[-2]):
        return (f"finished: stalled (M/eps {fl[-2]['M/eps first']:.4f} -> {fl[-1]['M/eps last']:.4f}, "
                f"< {rate:g} relative per solve hour in the last two segments)")
    return "continue"


def status(run_dir, n=8):
    ss = segs(run_dir)
    if not ss:
        print(f"{run_dir}: no segments yet")
        return
    seg = ss[-1]
    log_ = os.path.join(seg, "run.log")
    fin = os.path.join(run_dir, "FINISHED")
    print(f"== {os.path.basename(run_dir)} / {os.path.basename(seg)} (phase {seg_phase(seg) or '-'}); "
          f"{len(ss)} segment(s)" + (f"; FINISHED: {open(fin).read().strip()}" if os.path.exists(fin) else ""))
    if os.path.exists(os.path.join(run_dir, "chain.tsv")):
        print("-- chain.tsv (finished segments)")
        print(open(os.path.join(run_dir, "chain.tsv")).read().rstrip())
    if not os.path.exists(log_):
        print("no run.log yet")
        return
    age = time.time() - os.path.getmtime(log_)
    lines = open(log_, errors="replace").read().splitlines()
    steps = [l for l in lines if STEP_RE.match(l)]
    print(f"-- live log {log_}: last write {age / 60:.1f} min ago, {len(lines)} lines, {len(steps)} inner steps")
    setup = [l for l in lines if l.startswith("[setup]")]
    if setup and not steps:
        print("   in setup: " + setup[-1][:160])
    vals = [STEP_RE.match(l) for l in steps]
    if vals:
        me = [float(v.group(10)) for v in vals]
        ts = np.cumsum([float(v.group(11)) for v in vals]) / 3600
        rat = [float(v.group(4)) for v in vals]
        hit = sum(1 for v in vals if v.group(9))
        print(f"   M/eps {me[0]:.4f} -> {me[-1]:.4f} over {ts[-1]:.2f} h of steps "
              f"({(me[0] - me[-1]) / me[0] / max(ts[-1], 1e-9):.2e} relative per hour); "
              f"boundary hits {hit}/{len(vals)}; ratio < 0.1 on {sum(r < 0.1 for r in rat)} steps")
        if len(me) >= 10:
            k = len(me) // 2
            print(f"   second half: {me[k]:.4f} -> {me[-1]:.4f} "
                  f"({(me[k] - me[-1]) / me[k] / max(ts[-1] - ts[k], 1e-9):.2e} per hour)")
    for tag in ("[diag]", "[metric]", "[subspace]", "[noise]", "[control]", "stall detector", "alpha "):
        hits = [l for l in lines if tag in l]
        if hits:
            print(f"   last {tag.strip()}: {hits[-1].strip()[:200]}" + (f"  ({len(hits)} total)" if len(hits) > 1 else ""))
    bad = [l for l in lines if re.search(r"Traceback|Error|RESOURCE_EXHAUSTED|out of memory|\bnan\b", l)]
    if bad:
        print("!! problems: " + " | ".join(b.strip()[:120] for b in bad[-3:]))
    if age > 3600:
        print(f"!! no log output for {age / 60:.0f} min: check the job (squeue) and GPU (gpumem.txt)")
    print(f"-- last {n} step lines")
    for l in steps[-n:]:
        print(l)


def report(runs, names):
    out = os.path.join(runs, "report")
    os.makedirs(out, exist_ok=True)
    md = ["# Long-run report", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')} from {runs}.", ""]
    for name in names:
        rd = os.path.join(runs, name)
        rows, drows, t_off = [], [], 0.0
        for s in segs(rd):
            run = seg_json(s) or {}
            for a, h in _hists(run):
                t_off += h.get("time", 0.0)
                if h.get("accepted"):
                    rows.append((os.path.basename(s), seg_phase(s), a, t_off / 3600, 1 + h["c"][0], (1 + h["c"][0]) * a,
                                 h.get("ratio"), bool(h.get("hit")), h.get("lanczos")))
            for d in run.get("diag log", []):
                drows.append((os.path.basename(s), seg_phase(s), d.get("where"), d.get("M/M_desc"), d.get("|J-J_fit|/rms"),
                              d.get("b/vac"), d.get("b phase deg"), d.get("band p16/DESC"), d.get("band max/DESC"),
                              d.get("outside mean/DESC")))
        with open(os.path.join(out, f"steps_{name}.csv"), "w") as fh:
            fh.write("segment,phase,alpha,solve_h_cum,M_over_eps,M_over_Mdesc,ratio,hit,lanczos\n")
            fh.writelines(",".join(str(x) for x in r) + "\n" for r in rows)
        with open(os.path.join(out, f"diag_{name}.csv"), "w") as fh:
            fh.write("segment,phase,where,M_over_Mdesc,dJ_rel,b_over_vac,b_phase_deg,band_p16_over_DESC,"
                     "band_max_over_DESC,outside_mean_over_DESC\n")
            fh.writelines(",".join(f'"{x}"' if isinstance(x, str) else str(x) for x in r) + "\n" for r in drows)
        md += [f"## {name}", ""]
        if os.path.exists(os.path.join(rd, "FINISHED")):
            md += [f"Chain status: {open(os.path.join(rd, 'FINISHED')).read().strip()}", ""]
        if os.path.exists(os.path.join(rd, "chain.tsv")):
            t = [l.split("\t") for l in open(os.path.join(rd, "chain.tsv")).read().splitlines()]
            md += ["| " + " | ".join(t[0]) + " |", "|" + "---|" * len(t[0])] + ["| " + " | ".join(r) + " |" for r in t[1:]] + [""]
        if rows:
            md += [f"Accepted steps: {len(rows)} over {rows[-1][3]:.1f} h of solve time; M/M_DESC "
                   f"{rows[0][5]:.5f} -> {rows[-1][5]:.5f} (solve grid).", ""]
        if drows:
            f0, f1 = drows[0], drows[-1]
            md += [f"Diagnostics ({len(drows)} records): b/vac {f0[5]:.4f} -> {f1[5]:.4f} (phase {f1[6]:+.1f} deg); "
                   f"band p16/DESC {f0[7]:.4f} -> {f1[7]:.4f}; outside mean/DESC {f0[9]:.4f} -> {f1[9]:.4f}; "
                   f"|J-J_fit|/rms {f0[4]:.4e} -> {f1[4]:.4e}.", ""]
        try:  # optional figure
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            if rows:
                fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
                T = [r[3] for r in rows]
                ax[0].plot(T, [r[5] for r in rows], drawstyle="steps-post")
                ax[0].set_xlabel("cumulative solve time (h)")
                ax[0].set_ylabel("M / M_DESC (solve grid)")
                ax[0].set_yscale("log")
                if drows:
                    ax[1].plot(range(len(drows)), [d[5] for d in drows], "o-", ms=3)
                    ax[1].set_xlabel("diagnostic record")
                    ax[1].set_ylabel("b21 / vacuum")
                fig.suptitle(name)
                fig.tight_layout()
                fig.savefig(os.path.join(out, f"{name}.png"), dpi=150)
                md += [f"![{name}]({name}.png)", ""]
        except Exception as e:  # noqa: BLE001
            md += [f"(no figure: {e!r})", ""]
    open(os.path.join(out, "report.md"), "w").write("\n".join(md) + "\n")
    print(os.path.join(out, "report.md"))


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "newest-state":
        print(newest_state(args[0], args[1] if len(args) > 1 else None))
    elif cmd == "strip":
        strip(args[0], args[1])
    elif cmd == "remaining-alphas":
        print(" ".join(remaining_alphas(args[0], args[1:])))
    elif cmd == "summarize":
        summarize(args[0])
    elif cmd == "decide":
        print(decide(args[0], float(args[1]), int(args[2])))
    elif cmd == "status":
        status(args[0], int(args[1]) if len(args) > 1 else 8)
    elif cmd == "report":
        report(args[0], args[1:])
    else:
        raise SystemExit(__doc__)
