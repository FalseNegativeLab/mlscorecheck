Requirements
************

In general, there are three inputs to the consistency testing functions:

* **the specification of the experiment**;
* **the collection of available (reported) performance scores**: when aggregated performance scores (averages on folds or datasets) are reported, only accuracy (``acc``), sensitivity (``sens``), specificity (``spec``) and balanced accuracy (``bacc``) are supported; when cross-validation is not involved in the experimental setup, the list of supported scores reads as follows (with abbreviations in parentheses):

  * accuracy (``acc``),
  * sensitivity (``sens``),
  * specificity (``spec``),
  * positive predictive value (``ppv``),
  * negative predictive value (``npv``),
  * balanced accuracy (``bacc``),
  * f1(-positive) score (``f1``),
  * f1-negative score (``f1n``),
  * f-beta positive (``fbp``),
  * f-beta negative (``fbn``),
  * Fowlkes-Mallows index (``fm``),
  * unified performance measure (``upm``),
  * geometric mean (``gm``),
  * markedness (``mk``),
  * positive likelihood ratio (``lrp``),
  * negative likelihood ratio (``lrn``),
  * Matthews correlation coefficient (``mcc``),
  * bookmaker informedness (``bm``),
  * prevalence threshold (``pt``),
  * diagnostic odds ratio (``dor``),
  * Jaccard index (``ji``),
  * Cohen's kappa (``kappa``);

* **the estimated numerical uncertainty**: the performance scores are usually shared with some finite precision, being rounded/ceiled/floored to ``k`` decimal places. The numerical uncertainty estimates the maximum difference of the reported score and its true value. For example, having the accuracy score 0.9489 published (4 decimal places), one can suppose that it is rounded, therefore, the numerical uncertainty is 0.00005 (10^(-4)/2). To be more conservative, one can assume that the score was ceiled or floored. In this case, the numerical uncertainty becomes 0.0001 (10^(-4)).
