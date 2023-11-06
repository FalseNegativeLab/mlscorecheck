Requirements
************

In general, there are three inputs to the consistency testing functions:

* **the specification of the experiment**;
* **the collection of available (reported) performance scores**;
* **the estimated numerical uncertainty**: the performance scores are usually shared with some finite precision, being rounded/ceiled/floored to ``k`` decimal places. The numerical uncertainty estimates the maximum difference of the reported score and its true value. For example, having the accuracy score 0.9489 published (4 decimal places), one can suppose that it is rounded, therefore, the numerical uncertainty is 0.00005 (10^(-4)/2). To be more conservative, one can assume that the score was ceiled or floored. In this case, the numerical uncertainty becomes 0.0001 (10^(-4)).
