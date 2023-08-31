Requirements
************

In general, there are three inputs to the consistency testing functions:

* the specification of the dataset(s) involved;
* the collection of available performance scores. The currently supported scores with their abbreviations in paranthesis are:

  * accuracy (``acc``),
  * sensitivity (``sens``),
  * specificity (``spec``),
  * balanced accuracy (``bacc``)
  * positive predictive value (``ppv``),
  * negative predictive value (``npv``),
  * F1-score (``f1``),
  * Fowlkes-Mallows index (``fm``);
* the estimated numerical uncertainty: the performance scores are usually shared with some finite precision, being rounded/ceiled/floored to ``k`` decimal places. The numerical uncertainty estimates the maximum difference of the reported score and its true value. For example, having the accuracy score 0.9489 published (4 decimal places), one can suppose that it is rounded, therefore, the numerical uncertainty is 0.00005 (10^(-4)/2). To be more conservative, one can assume that the score was ceiled or floored. In this case the numerical uncertainty becomes 0.0001 (10^(-4)).
