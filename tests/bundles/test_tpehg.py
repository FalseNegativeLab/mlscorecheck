"""
Testing the EHG testing
"""

from mlscorecheck.bundles.ehg import check_tpehg

def test_check_tpehg():
    """
    Testing the check_tpehg function
    """

    results = check_tpehg(scores={'acc': 0.95, 'sens': 0.95, 'spec': 0.95},
                        eps=1e-4,
                        n_folds=5,
                        n_repeats=1)

    assert 'inconsistency' in results
