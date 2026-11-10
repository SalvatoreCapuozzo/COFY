# Paper draft — Discover Artificial Intelligence, "Trustworthy and Responsible Federated Learning"

Deadline: **15 December 2026**. Template: Springer Nature LaTeX template (December 2024,
`sn-jnl.cls`, numbered references `sn-mathphys-num`), as recommended by the publisher.
The journal asks for an abstract under 250 words (now 249).

    latexmk -pdf main.tex supplementary.tex   # main.pdf, then supplementary.pdf (TeX Live 2022)

`supplementary.tex` is the Supplementary Information, submitted as a separate PDF. It reads the
main article's section and table numbers from `main.aux` (package `xr-hyper`), so build `main.tex`
first. The main article does not mention it (author decision).

## Markers in the text

- red `\pending{...}`: none left (2026-11-09). Sources of the results added last:
  - robustness and backdoor success rate: `analysis/ROBUSTNESS.md` (`python tools/robust_eval.py --report`);
  - competitors: `python tools/results_table.py` (rows `bl_*`);
  - local-only baseline, averaged centralized reference, per-class budget: `tools/followup_runs.py`
    (summaries in `runs/variants/{local_only,central_swa,rb_budget_clean}/`, figures in
    `analysis/FOLLOWUP_PLAN.md` section 12).
- The confirmation is complete (2026-11-04): `analysis/CONFIRMATORY_RESULTS.md`
  (`python tools/confirm_analysis.py --write`), fairness `python tools/fairness_analysis.py --confirm`,
  cost `python tools/cost_report.py`.
- blue `\authornote{...}`: decisions for the authors (e-mails, the registered verdict of the
  concurrent scheme, how the third registered use case is presented, kfm12 release DOI, CRediT,
  release tag).

## Structure

1 Introduction (ends with the outline) - 2 Related work - 3 Materials and methods (problem setting,
the COFY system, deployment and defence, use cases) - 4 Experimental design - 5 Results (cost of
federation, fairness and cost, standard methods and local-only baseline, robustness) - 6 Discussion -
7 Conclusions - Declarations - Appendix A (reproducibility). At most four subsections per section,
each section opens with an introduction.

## References

`references.bib` holds only works identified with confidence. Entries with a `verify` field
have details (authors, pages, final venue) that were not checked against the publisher's page
while drafting. Check them before submission, as well as every entry's DOI.
