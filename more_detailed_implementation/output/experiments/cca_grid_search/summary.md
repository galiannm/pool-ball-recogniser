# CCA recognition search

Search base: frozen configuration 2153 (K=6, light V>=170/S<=125, cue>=85%, solid>=80%, blue/purple hue boundary 125), no padding, no cloth filter.

Tuning: Table1-6. Reference only: AllBalls. These are exploratory tuning results.

33124 trials including four controls; 12 feature settings, all eight rule masks, two passes of grouped cue/stripe/solid threshold searches. Repeated configurations are retained.

Rule mask bits: cue=1, stripe=2, solid=4. Rule order: cue, stripe, solid, baseline fallback.

Ranking: correct numbers, correct types, fewer number regressions, fewer type regressions, fewer changed types, fewer enabled rules, smaller departure from original CCA parameters, earlier ID. The best-type ranking swaps the first two criteria. Regression counts always compare with tuned K=6.

| Configuration | Type (tuning) | Number (tuning) | Type fixed/worsened | Number fixed/worsened | Type/number (reference) |
|---|---:|---:|---:|---:|---:|
|original_k2 (1)|76/86|67/86|0/5|0/7|16/16 (of 16)|
|original_k2_cca (2)|73/86|64/86|0/8|0/10|16/16 (of 16)|
|tuned_k6 (3)|81/86|74/86|0/0|0/0|16/16 (of 16)|
|tuned_k6_cca (4)|79/86|72/86|0/2|0/2|16/16 (of 16)|
|best_number (20242)|82/86|75/86|1/0|1/0|16/16 (of 16)|
|best_type (20242)|82/86|75/86|1/0|1/0|16/16 (of 16)|

Changing CCA only changes type and the downstream type-dependent number decision. It does not change cluster colours, body selection or hue thresholds. Small-component filtering affects component statistics, not raw light percentages or boundary coverage. K=6 features use the same light definition as configuration 2153; the original K=2 control uses the production light definition. Default features and hybrid predictions were checked against production on every candidate. Main application files remain unchanged.

With K=6 segmentation and number rules fixed, supplying the true type correctly numbers 75/86 tuning balls. This is the full-number ceiling for type-only changes on these cached candidates, not an upper bound for a different colour/extraction pipeline.
