"""
Test functions for the EHG problem
"""

from ..experiments import load_ehg

from ..check import check_1_dataset_unknown_folds_mor_scores

def check_ehg(scores: dict,
                eps,
                n_folds: int,
                n_repeats: int) -> dict:
    """
    Checks the cross-validated EHG scores

    Args:
        scores (dict(str,float)): the dictionary of scores
        eps (float|dict(str,float)): the numerical uncertainties
        n_folds (int): the number of folds
        n_repeats (int): the number of repetitions

    Returns:
        dict: the result of the consistency testing
    """
    evaluation = {'dataset': load_ehg(),
                    'folding': {'n_folds': n_folds, 'n_repeats': n_repeats}}

    return check_1_dataset_unknown_folds_mor_scores(scores=scores,
                                                    eps=eps,
                                                    evaluation=evaluation)
