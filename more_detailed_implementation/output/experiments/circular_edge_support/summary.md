# Circular edge support diagnostic

Measurement only: no circle is rejected, moved or resized. Settings are fixed before scoring.

72 boundary directions (5 degrees apart); search within 2.5 source-image pixels; Canny 37.5/75 after 5x5 Gaussian blur, sigma 1.5; absolute radial gradient alignment >= 0.6. Polarity is ignored. Edges come from the original image, and support is restricted to the table.

Support is aligned supported directions / valid directions. Coverage is valid / 72. Raw longest arc accounts for wrap-around. Bridged arc also fills isolated missing directions; it does not change the support fraction. Nearby-edge fraction ignores alignment. These scores are geometric evidence, not calibrated probabilities.

## False detections and score overlap

Counts below compare each false detection to true detections in Table1–6 (AllBalls excluded). They diagnose score overlap on already inspected images; they do not select or validate a threshold.

| Image | Candidate | Support | Raw arc | Bridged arc | True detections with equal/lower support |
|---|---:|---:|---:|---:|---:|
| Table1.jpeg | 15 | 33.3% | 110.000000 | 110.000000 | 0/86 |
| Table2.jpeg | 17 | 45.8% | 165.000000 | 165.000000 | 1/86 |

Baseline Table1–6: 86 matched detections, 2 false detections.

## Lowest-scoring genuine detections

| Image | Candidate | Ball | Support | Coverage | Raw arc |
|---|---:|---:|---:|---:|---:|
| Table1.jpeg | 10 | 5 | 41.7% | 100.0% | 60.000000 |
| Table1.jpeg | 5 | 13 | 50.0% | 100.0% | 165.000000 |
| Table4.jpeg | 13 | 2 | 50.0% | 100.0% | 150.000000 |
| Table5.jpeg | 14 | 10 | 51.4% | 100.0% | 185.000000 |
| Table6.jpeg | 13 | 10 | 52.8% | 100.0% | 165.000000 |
| Table1.jpeg | 13 | 8 | 55.6% | 100.0% | 65.000000 |
| Table6.jpeg | 8 | 2 | 55.6% | 100.0% | 200.000000 |
| Table5.jpeg | 7 | 0 | 58.5% | 90.3% | 170.000000 |
| Table5.jpeg | 13 | 2 | 58.5% | 90.3% | 145.000000 |
| Table2.jpeg | 16 | 8 | 59.7% | 100.0% | 130.000000 |
| Table5.jpeg | 5 | 9 | 59.7% | 100.0% | 215.000000 |
| Table3.jpeg | 11 | 4 | 62.3% | 95.8% | 105.000000 |

Open the false-candidate PNGs at this directory's top level first. Each image folder contains one PNG per candidate, an overview (IDs and support percentages; green TP, red FP), and the Canny map. Close-up boundary colours mean green aligned, orange nearby but unaligned, red no edge, grey invalid; cyan dots mark supporting edge pixels. TP/FP labels come solely from one-to-one annotation matching within 15 table pixels.

Touching balls, shadows, number patches and edges shared with neighbours can all affect support. Inspect close-ups alongside scores before designing a rejection rule. The main detector, recognition and report are unchanged.
