Retinal vessel segmentation
===========================

The segmentation of the vasculature in retinal images gained enormous attention in the recent decade. Recently, using the tools implemented in this package, it was shown in [RV]_ that the authors use at least two different evaluation techniques, leading to incomparable performance scores and skewing the field. Retinal images have valueable image content only in the circular Field-of-View (FoV) region:

.. image:: https://rumc-gcorg-p-public.s3.amazonaws.com/i/2020/01/21/75d803ef.png

When a binary segmentation of the vasculature is evaluated, some authors do calculate the performance scores only in the FoV region, others involve also the pixels outside the FoV region, which increases the number of true negatives (and consequently, accuracy and specificity) enormously. The functionalities implemented in this package are suitable to distinguish the two kinds of evaluations based on the reported performance scores.

The package contains the statistics of the images of the DRIVE dataset, and provides the functionality of checking the consistency of reported scores with the assumptions of using the FoV pixels only or using all pixels for evaluation, for both image level figures and score aggregated for all images of the DRIVE dataset.

Again, we highlight, that the techniques in the package detect inconsistency with certainty. If the use of the FoV pixels only was found to be inconsistent with a certain set of scores, the user can conclude that these scores are not comparable with other scores which are calculated in the FoV region.

For example, given a set of performance scores at the test image level and the claim of evaluating in the FoV region only, one can execute the code block below to validate the claim.

.. code-block:: Python

    >>> result = drive_image_fov_pixels(scores={'acc': 0.9478, 'npv': 0.8532,
                                            'f1p': 0.9801, 'ppv': 0.8543},
                                    eps=1e-4,
                                    image_set='test',
                                    identifier='01')
    >>> result['inconsistency']
    # True

Similarly, one can check if the scores are inconsistent with the assumption of using all pixels for for evaluation:

.. code-block:: Python

    >>> result = drive_image_all_pixels(scores={'acc': 0.9478, 'npv': 0.8532,
                                            'f1p': 0.9801, 'ppv': 0.8543},
                                    eps=1e-4,
                                    image_set='test',
                                    identifier='01')
    >>> result['inconsistency']
    # True

If the way of evaluation is not specified, one can still use the functionalities provided in the package to check if the scores are consistent with any of the two reasonable assumptions:

.. code-block:: Python

    >>> drive_image(scores={'acc': 0.9478, 'npv': 0.8532,
                                'f1p': 0.9801, 'ppv': 0.8543},
                        eps=1e-4,
                        image_set='test',
                        identifier='01')
    # {'fov_inconsistency': True, 'no_fov_inconsistency': True}

It is common that image level performance scores are not reported, but only aggregations (like average accuracy on 20 images, etc.). The consistency of the aggregated figures with the assumption of using FoV pixels only can be tested as:

.. code-block:: Python

    >>> result = drive_aggregated_fov_pixels(scores={'acc': 0.9478,
                                                        'sens': 0.8532,
                                                        'spec': 0.9801},
                                            eps=1e-4,
                                            image_set='test')
    >>> result['inconsistency']
    # True

Similarly, the assumption of using all pixels can be tested by:

.. code-block:: Python

    >>> result = drive_aggregated_all_pixels(scores={'acc': 0.9478,
                                                        'sens': 0.8532,
                                                        'spec': 0.9801},
                                            eps=1e-4,
                                            image_set='test')
    >>> result['inconsistency']
    # True

Finally, if the both assumptions can be tested as

.. code-block:: Python

    >>> drive_aggregated(scores={'acc': 0.9478, 'sens': 0.8532, 'spec': 0.9801},
                        eps=1e-4,
                        image_set='test')
    # {'mor_fov_pixels_inconsistency': True,
    #    'mor_all_pixels_inconsistency': True,
    #    'rom_fov_pixels_inconsistency': True,
    #    'rom_all_pixels_inconsistency': True}
