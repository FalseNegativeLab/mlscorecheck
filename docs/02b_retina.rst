Retinal vessel segmentation
===========================

The segmentation of the vasculature in retinal images gained enormous attention in the recent decade. Recently, using the tools implemented in this package, it was shown in [RV]_ that the authors use at least two different evaluation techniques, leading to incomparable performance scores and skewing the field. Retinal images have valueable image content only in the circular Field-of-View (FoV) region:

.. image:: https://rumc-gcorg-p-public.s3.amazonaws.com/i/2020/01/21/75d803ef.png

When a binary segmentation of the vasculature is evaluated, some authors do calculate the performance scores only in the FoV region, others involve also the pixels outside the FoV region, which increases the number of true negatives (and consequently, accuracy and specificity) enormously. The functionalities implemented in this package are suitable to distinguish the two kinds of evaluations based on the reported performance scores.

The package contains the statistics of the images of the DRIVE dataset, and provides the functionality of checking the consistency of reported scores with the assumptions of using the FoV pixels only or using all pixels for evaluation, for both image level figures and score aggregated for all images of the DRIVE dataset.

Again, we highlight, that the techniques in the package detect inconsistency with certainty. If the use of the FoV pixels only was found to be inconsistent with a certain set of scores, the user can conclude that these scores are not comparable with other scores which are calculated in the FoV region.

The first function enables the testing of performance scores reported for certain test images, the two tests executed assume the use of the FoV mask (excluding the pixels outside the FoV) and the neglection of the FoV mask (including the pixels outside the FoV). As the following example shows, one simply supplies the scores and specifies the images (whether it is from the 'test' or 'train' subset and the identifier of the image) and gets back if inconsistency is identified with any of the two assumptions.

.. code-block:: Python

    >>> from mlscorecheck.bundles import (drive_image, drive_aggregated)

    >>> drive_image(scores={'acc': 0.9478, 'npv': 0.8532, 'f1p': 0.9801, 'ppv': 0.8543},
                    eps=1e-4,
                    bundle='test',
                    identifier='01')
    # {'fov_inconsistency': True, 'no_fov_inconsistency': True}

The interpretation of these results is that the reported scores are inconsistent with any of the reasonable evaluation methodolgoies.

A similar functionality is provided for the aggregated scores calculated on the DRIVE images, in this case the two assumptions of using the pixels outside the FoV is extended with two assumptions on the way of aggregation.

.. code-block:: Python

    >>> drive_aggregated(scores={'acc': 0.9478, 'sens': 0.8532, 'spec': 0.9801},
                        eps=1e-4,
                        bundle='test')
    # {'mor_fov_inconsistency': True,
    #   'mor_no_fov_inconsistency': True,
    #   'rom_fov_inconsistency': True,
    #   'rom_no_fov_inconsistency': True}

The results here show that the reported scores could not be the result of any aggregation of any evaluation methodologies.
