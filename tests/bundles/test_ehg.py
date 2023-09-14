"""
Testing the EHG testing
"""

from mlscorecheck.bundles import check_ehg

def test_check_ehg():
    """
    Testing the check_ehg function
    """

    results = check_ehg(scores={'acc': 0.95, 'sens': 0.95, 'spec': 0.95},
                        eps=1e-4,
                        n_folds=5,
                        n_repeats=1)

    assert 'inconsistency' in results
