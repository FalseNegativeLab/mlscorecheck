Retinal vessel segmentation
---------------------------

The segmentation of the vasculature in retinal images [RV]_ gained enormous interest in the recent decades. Typically, the authors have the option to include or exclude certain parts of the images (the pixels outside the Field-of-View), making the reported scores incomparable. (For more details see [RV]_.) To facilitate the meaningful comparison, evaluation and interpretation of reported scores, we provide two functions to check the internal consistency of scores reported for the DRIVE retinal vessel segmentation dataset.

The first function enables the testing of performance scores reported for specific test images. Two tests are executed, one assuming the use of the FoV mask (excluding the pixels outside the FoV) and the other assuming the neglect of the FoV mask (including the pixels outside the FoV). As the following example illustrates, one simply provides the scores and specifies the image (whether it is from the 'test' or 'train' subset and the image identifier) and the consistency results with the two assumptions are returned.

.. code-block:: Python

    >>> from mlscorecheck.bundles import (drive_image, drive_aggregated)

    >>> drive_image(scores={'acc': 0.9478, 'npv': 0.8532, 'f1p': 0.9801, 'ppv': 0.8543},
                    eps=1e-4,
                    bundle='test',
                    identifier='01')
    # {'fov_inconsistency': True, 'no_fov_inconsistency': True}

The interpretation of these results is that the reported scores are inconsistent with any of the reasonable evaluation methodologies.

A similar functionality is provided for the aggregated scores calculated on the DRIVE images, in this case the two assumptions of using the pixels outside the FoV is extended with two assumptions on the way of aggregation.

.. code-block:: Python

    >>> drive_aggregated(scores={'acc': 0.9478, 'sens': 0.8532, 'spec': 0.9801},
                        eps=1e-4,
                        bundle='test')
    # {'mos_fov_inconsistency': True,
    #   'mos_no_fov_inconsistency': True,
    #   'som_fov_inconsistency': True,
    #   'som_no_fov_inconsistency': True}

The results here show that the reported scores could not be the result of any aggregation of any evaluation methodologies.
