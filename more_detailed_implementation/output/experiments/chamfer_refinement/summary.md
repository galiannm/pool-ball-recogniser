# White-candidate chamfer boundary refinement

All four profiles use the same frozen AllBalls histogram and best-effort uniqueness assignment. Only original K2 White crops are modified. Detection counts, stored Hough circles and table coordinates remain fixed.

| Method | Correct identities | Correct types | Corrected vs C | Regressed vs C | Modified crops |
|---|---:|---:|---:|---:|---:|
| C_unique | 75/86 | 76/86 | 0 | 0 | 0 |
| white_pad1_unique | 79/86 | 79/86 | 4 | 0 | 9 |
| chamfer_minimum_unique | 75/86 | 76/86 | 0 | 0 | 3 |
| chamfer_outer_unique | 77/86 | 78/86 | 2 | 0 | 6 |

Yellow outlines show original Hough circles; cyan outlines show recognition crop circles. The second row shows eligible coloured-body pixels for each recognition crop. The crop comparison sheets contain predictions, not ground-truth labels. Use predictions.csv and changes.csv for automatic correctness.

The oriented chamfer score measures edge alignment, not identity confidence. Inner white patches can also have strong circular edges; shape evidence alone cannot guarantee the physical ball boundary. Neighbour edges, perspective, shadows and incomplete arcs remain limitations. The outer variant adds a fixed largest-supported-radius preference; it is not a parameter search or a validated optimum. Refined centre localisation is reported separately in refined_centre_metrics.csv; final_results intentionally preserve original detection geometry.

In this run, the outer-supported variant corrects Table1's cue/14 swap. The actual adjustments are small: the real cue shifts down1 pixel with radius unchanged24.15, reducing eligible body pixels75 to42; ball14 shifts down1 pixel and grows17.55 to18.55, yielding15 eligible body pixels. Radius+1 alone yields only8, below the frozen minimum12. Thus the improvement involves centre/mask discretisation and the body-pixel cutoff as well as edge evidence; it does not establish recovery of an entire physical outer boundary. The real cue's green rim can include cloth, which gives the histogram misleading green evidence. The minimum-cost variant changes3 crops without correcting identities; the outer preference changes6 and corrects2.

Parameters were fixed before evaluation, but this experiment follows observations on Table1–6 and is exploratory. No production integration or report modification is performed.
