"""
This module imports the solutions
"""

from io import StringIO

from importlib.resources import files

import json

from ._solver import Solutions

__all__ = ['load_solutions',
            'load_scores']

def load_solutions():
    sio = files('mlscorecheck').joinpath('core/solutions.json').read_text() # pylint: disable=unspecified-encoding

    solutions = json.loads(sio)

    results = {}

    for sol in solutions['solutions']:
        scores = [score['abbreviation'] for score in sol['scores']]
        results[tuple(sorted(scores))] = Solutions(**sol)

    return results

def load_scores():
    sio = files('mlscorecheck').joinpath('core/scores.json').read_text() # pylint: disable=unspecified-encoding

    scores = json.loads(sio)

    return scores['scores']