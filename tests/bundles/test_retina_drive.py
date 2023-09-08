"""
Testing the retina drive check bundle
"""

import pytest

from mlscorecheck.bundles import (drive_aggregated,
                                    drive_image,
                                    filter_drive)

def test_filter():
    """
    Testing the filtering function
    """

    data = [{'identifier': '01'},
            {'identifier': '02'},
            {'identifier': '03'}]

    assert len(filter_drive(data, ['01', '02'])) == 2
    assert filter_drive(data) == data

    with pytest.raises(ValueError):
        filter_drive(data, ['dummy'])

def test_aggregated():
    """
    Testing the aggregated check
    """
    results = drive_aggregated({'acc': 0.950, 'sens': 0.899, 'spec': 0.9834},
                                eps=1e-4,
                                image_set='test')

    assert len(results) == 4

def test_image():
    """
    Testing the image check
    """
    results = drive_image({'acc': 0.950, 'sens': 0.899, 'spec': 0.9834},
                                eps=1e-4,
                                image_set='test',
                                identifier='01')

    assert len(results) == 2
