# Fixed prototype parameters

See experiments/watershedDetection.h for defaults. No parameter grid or annotation-driven marker selection is used.

- Nearby-circle grouping: distance <= 1.4 times sum of radii; at least two detections.
- Crop padding: 2 median radii; overlapping crops merge.
- Cloth: HSV H 35-95, S>=35, V>=30; sample stride 2; exclude disks of 1.25 Hough radii.
- Lab: per-channel median cloth; threshold = median distance + 2.5 MAD, clipped to [16,45].
- Require 40 cloth samples; keep known foreground cores of 0.65 Hough radius.
- Close with a 3x3 ellipse, fill enclosed holes, intersect with table polygon.
- Distance maxima: neighbourhood approximately 0.9 median radii wide; distance >=0.45 radius.
- Markers: radius 3; separation >=1.15 times the larger radius; crop edge margin 0.6 radius.
- Sure background: outside a 7x7 dilation of foreground; unknown collar inside it.
- Watershed image: original BGR crop smoothed with 3x3 Gaussian, sigma=0.8.
- Basin area / expected disk area: [0.25,1.8]; circularity >=0.35; aspect ratio <=2.2.
- Foreground coverage of expected disk >=0.35; existing pocket filter applies to additions.
- New centre: distance peak, not basin centroid. New radius: local median Hough radius.
- Recognition: configuration 2153 plus selected solid-only CCA (2% cutoff, 5x5 band, non-light>=70%, light boundary<=20%, exactly one non-light component). Standard circular masks are used for all methods; watershed masks are diagnostic only.
