"""
This module implements consistency tests at the individual dataset level
"""

__all__ = ['consistency_individual']

def consistency_individual(p, n, acc, sens, spec, eps):
    """
    Checking the consistency of individual figures
    
    Args:
        p (int): assumed number of positives
        n (int): assumed number of negatives
        acc (float): the observed accuracy score
        sens (float): the observed sensitivity score
        spec (float): the observed specificity score
        eps (float): the assumed +/- numerical uncertainty of the observed figures
    
    Returns:
        boolean: True if the observed scores are consistent with the figures, False otherwise
    """
    term0 = (n*(acc - spec) + p*(acc - sens) - 2*eps*(p+n)) <= 0
    term1 = 0 <= (n*(acc - spec) + p*(acc - sens) + 2*eps*(p + n))
    term2 = 0 >= p*(sens - eps - 1)
    term3 = 0 <= p*(sens + eps)
    term4 = 0 >= n*(spec - eps - 1)
    term5 = 0 <= n*(spec + eps)

    return term0 and term1 and term2 and term3 and term4 and term5

