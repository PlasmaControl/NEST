# results/ripple_spline

Data of the ripple-landscape rerun with B-spline representations (branch `ripple-spline`). Reports:
`docs/ripple_spline_report.md` (FZ vs FZ + band splines), `docs/ripple_spline_findings.md` (all findings, sections
1-16), `docs/ripple_spline_plan.md` (plan and run log).

| path | content |
|---|---|
| `summary.md`, `summary.json` | `scripts/ripple_spline/collect.py`: beta scan, releases, islands, effort for report_fz / fzc / h1 |
| `common/` | states evaluated on the common grid (a/120, the hybrid's radial collocation): report_fz, fzc, h1, fz40, h2, h3, h4 |
| `states/h1`, `states/fzc` | shielded starts, pinned-beta states (`b<beta>.npz`; `_first` = before the extension), releases, seeds |
| `fitsplit/` | `--fit-split`, `--wfit` (exact weighted J fits), `res_*` (standard spline fit at higher resolution) |
| `phase0/` | fixed-J pressure tests (spline / hybrid pressure on the report's FZ states) |

Not included (too large, ~95 MB): the run records (`<run>.json`) and live logs (`<run>_live.tsv`) of every run. They are
on Stellar-AI in `/scratch/gpfs/EKOLEMEN/lk1158/NEST_rspl/out/rs/<run>/` (h1, fzc, fz40, h2, h3, h4, s1, s2, s1w, s2w).
