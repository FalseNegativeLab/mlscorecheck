"""
This module tests the functionalities related to problem generation
"""

from mlscorecheck.aggregated import generate_scores_for_testsets

def test_generate_scores_for_testsets():
    """
    Testing the generation of scores for multiple testsets
    """

    scores = generate_scores_for_testsets([{'p': 5, 'n': 10, 'identifier': 'a'},
                                            {'p': 6, 'n': 20, 'identifier': 'b'}],
                                            rounding_decimals=4,
                                            subset=['acc', 'sens'],
                                            aggregation='mos')
    assert len(scores) == 2

    scores = generate_scores_for_testsets([{'p': 5, 'n': 10, 'identifier': 'a'},
                                            {'p': 6, 'n': 20, 'identifier': 'b'}],
                                            rounding_decimals=4,
                                            subset=['acc', 'sens'],
                                            aggregation='som')

    assert len(scores) == 4
