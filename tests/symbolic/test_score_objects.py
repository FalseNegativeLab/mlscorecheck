"""
Testing the score objects
"""

import pytest

from mlscorecheck.symbolic import (
    get_base_objects,
    get_symbolic_toolkit,
    get_all_objects,
    get_objects_without_complements,
    Score,
)

symbolic_toolkit = get_symbolic_toolkit()


@pytest.mark.skipif(symbolic_toolkit is None, reason="no symbolic toolkit available")
def test_base_objects():
    """
    Testing the instantiation of the base objects
    """
    assert len(get_base_objects(symbolic_toolkit)) > 0


@pytest.mark.skipif(symbolic_toolkit is None, reason="no symbolic toolkit available")
def test_all_objects():
    """
    Testing the instantiation of the base objects
    """
    assert len(get_all_objects(symbolic_toolkit)) > 0


@pytest.mark.skipif(symbolic_toolkit is None, reason="no symbolic toolkit available")
def test_no_complement_objects():
    """
    Testing the instantiation of the base objects
    """
    assert len(get_objects_without_complements(symbolic_toolkit)) > 0


@pytest.mark.skipif(symbolic_toolkit is None, reason="no symbolic toolkit available")
def test_serialization():
    """
    Testing the serialization
    """
    base_objects = get_base_objects(symbolic_toolkit)

    serialized = base_objects["acc"].to_dict()

    score = Score(symbols=base_objects["acc"].symbols, **serialized)

    assert score is not None
    assert score.get_algebra() == base_objects["acc"].symbols.algebra
