# Row/radius figure guide

Recommended main figure: **row_radius_samples** links the measurements to the source image, and **row_radius_fit** shows what the model learns and where it extrapolates. Use **row_radius_estimation** when space permits a detailed illustration of the robust estimator.

Sample figure: Original AllBalls image and enlarged detected-ball region. Yellow outlines show measured circles, crosses show centres, and numbers identify detections in training_samples.csv. Horizontal guides show original-image rows; the shaded overview band spans the observed centre rows.

Fit figure: Detected radii against centre row, with a robust linear fit. The second panel marks the observed training row range and distinguishes extrapolation using dashed lines. The line is estimated from detected circles, rather than measured physical radii.

Estimator figure: Distribution of 71 pairwise slopes from pairs separated by at least ten rows, with median b=0.009677441; individual intercept estimates with median a=20.274190. Figure reconstruction uses the six-decimal saved sample CSV, so final digits may differ from the C++ model.

All figures are provided as 300-dpi PNG and vector PDF. No test-image labels or measurements are used. The fitted row band is narrow; extension to other depths is an assumption. The underlying figures contain no code or library names.
