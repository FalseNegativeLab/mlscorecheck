"""
Testing the test functionalities for the ISIC melanoma datasets
"""

from mlscorecheck.bundles import (check_isic2016,
                                    check_isic2017m,
                                    check_isic2017sk)

def test_isic2016():
    """
    Testing the ISIC2016 dataset
    """

    assert not check_isic2016(scores={'acc': 0.855,
                                    'sens': 0.507,
                                    'spec': 0.941},
                            eps=1e-3)['inconsistency']

def test_isic2017m():
    """
    Testing the ISIC2017 melanoma problem
    """

    assert not check_isic2016(scores={'acc': 0.748,
                                    'sens': 0.538,
                                    'spec': 0.799},
                            eps=1e-3)['inconsistency']

def test_isic2017sk():
    """
    Testing the ISIC2017 seborrheic keratosis problem
    """

    assert not check_isic2016(scores={'acc': 0.711,
                                    'sens': 0.8,
                                    'spec': 0.696},
                            eps=1e-3)['inconsistency']
