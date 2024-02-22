# pyCorrCA
Python version to run Correlated Component Analysis (https://www.parralab.org/corrca/) and should be compatible with pipeline in scikit-learn. The code was revised from https://github.com/renzocom/CorrCA

What are different:
* There is a real class now, not sure why it was missing, and should support scikit-learn format (fit, transform, and fit_transform)
* Previous version may return very different result from the MATLAB version, mainly due to eigen value decomposition. See the NOTE in pyCorrCA.py for explanation. The current implementation should be consistent; the directions (signs) of eigen vectors and numerical precisions may still not match.
* Some functions from the previous version were not ported but one should not find it difficult to add back.
