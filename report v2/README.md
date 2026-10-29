# Report v2

This report describes the selected **experimental** `pad1_row_largest_cue`
configuration (80/86 matched identities), not the older application baseline.
Production integration and consolidation into the assignment's single C++ file
remain separate work. The original `report/` is unchanged.

Build in this folder using `make`. Source is `main.tex`, output is `main.pdf`.
The document uses 12-point A4 text and 25.4 mm margins, with no title page.
All pages, references and AI acknowledgement count toward the eight-page limit.

The revised draft has six main-text pages and two appendix pages. It explains
the selected two-cluster approach without discussing the six-cluster search.
Appendix A contains final rectified outputs for all six tables, also supplied as
individual `assets/final_table1.png` through `final_table6.png`. Appendix B holds
the radius-measurement illustration, references and AI acknowledgement. Technical
terms are introduced through their role in this task, with examples of how the
local colour/type evidence and joint identity assignment interact.

Figures/tables are regenerated from saved outputs with `make assets`. This needs
Python, NumPy, Matplotlib and Python OpenCV for offline report graphics only.
Application/experimental processing remains C++/OpenCV. The figure script checks
the illustrative K2 light fractions against recorded C++ evidence. It does not
tune parameters or rerun evaluation. `provenance.json` records metric sources.

Classification uses HSV **histogram comparison**; backprojection is a diagnostic
image, not the decision score. Inference and ground-truth evaluation are kept
distinct. Test-informed selection is explicitly disclosed; 80/86 is not claimed
as independent held-out performance. No ground-truth error is asserted without
evidence. References and AI acknowledgement must be retained/reviewed for the
final submission; the acknowledgement covers the substantial AI-generated code,
experiments, report draft and graphics.
