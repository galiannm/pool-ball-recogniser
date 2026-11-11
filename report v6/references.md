# Sources for report v6

The manuscript uses the following four sources:

1. **K. Dawson-Howe, _CSU44053 Computer Vision course material_, Trinity College Dublin.**
   Course background for the vision methods. Project-specific thresholds,
   reference construction and results are described from the implementation and
   saved outputs, rather than attributed to a particular lecture slide.

2. **Scikit-learn Developers, _User Guide_, §1.7.4.3, “Radial basis function (RBF) kernel.”**
   [Official documentation](https://scikit-learn.org/stable/modules/gaussian_process.html#radial-basis-function-rbf-kernel).
   Supports the Gaussian distance fall-off exp(-d²/(2l²)). The report uses distance
   to a preferred type interval rather than the distance between two data points.
   That adaptation, the interval endpoints and width 0.10 are project heuristics.
   This is not a Gaussian-process classifier, a fitted probability model or a
   claim that the reference validates the chosen recognition settings.

3. **MathWorks, _Fuzzy Logic Toolbox documentation_, “Gaussian combination membership function.”**
   [Official documentation](https://www.mathworks.com/help/fuzzy/gauss2mf.html).
   Describes a membership function with value one between two boundaries and a
   Gaussian fall-off on either side. With equal widths, this is mathematically
   equivalent to the report's type-support function over light coverage [0,1].
   Scikit-learn supports the shared Gaussian expression; MathWorks supports the
   interval construction. The project implements the formula directly in C++.
   Its interval endpoints and width 0.10 are project choices.

4. **CP-Algorithms, “Hungarian Algorithm.”**
   [Article](https://cp-algorithms.com/graph/hungarian-algorithm.html).
   Explains one-to-one assignment, rectangular matrices and the equivalence of
   maximum-score and minimum-cost formulations. It supports the joint optimisation
   principle, not the accuracy of the project's appearance scores. The numerical
   example uses measured Table2 candidates 6 and 11; saved independent and unique
   predictions verify the reassignment from duplicate 4s to identities 4 and 12.

Gaussian-function web sources checked on 10 November 2026; the Hungarian source
was checked on 9 November 2026. All three web references are clickable in the PDF.
