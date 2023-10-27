Retina image processing
-----------------------

DRIVE dataset
^^^^^^^^^^^^^

The [DRIVE]_ dataset consists of 20 train and 20 test images for the segmentation of retinal vessels. The ground truth is provided as binary masks of the same size as the images. The masks are provided in the PNG format. The images are provided with the Field-of-View (FoV) masks, which are also binary masks of the same size as the images. The FoV masks are provided in the PNG format. The images are provided with the masks of the optic disc, which are also binary masks of the same size as the images. The optic disc masks are provided in the PNG format.

Typically, the authors have the option to include or exclude certain parts of the images (the pixels outside the Field-of-View), making the reported scores incomparable. (For more details see [RV]_.) To facilitate the meaningful comparison, evaluation and interpretation of reported scores, we provide two functions to check the internal consistency of scores reported for the DRIVE retinal vessel segmentation dataset.

The first function enables the testing of performance scores reported for specific test images. Two tests are executed, one assuming the use of the FoV mask (excluding the pixels outside the FoV) and the other assuming the neglect of the FoV mask (including the pixels outside the FoV). As the following example illustrates, one simply provides the scores and specifies the image (whether it is from the 'test' or 'train' subset and the image identifier) and the consistency results with the two assumptions are returned.

.. code-block:: Python

    >>> from mlscorecheck.bundles.retina import check_drive_vessel_image
    >>> scores = {'acc': 0.9633, 'sens': 0.7406, 'spec': 0.9849}
    >>> identifier = '01'
    >>> k = 4
    >>> results = check_drive_vessel_image(scores=scores,
                                            eps=10**(-k),
                                            image_identifier=identifier,
                                            annotator=1)
    >>> results['inconsistency']
    # {'inconsistency_fov': True, 'inconsistency_all': False}

The interpretation of these results is that the reported scores are inconsistent with any of the reasonable evaluation methodologies.

A similar functionality is provided for the aggregated scores calculated on the DRIVE images, in this case the two assumptions of using the pixels outside the FoV is extended with two assumptions on the way of aggregation.

.. code-block:: Python

    >>> from mlscorecheck.bundles.retina import check_drive_vessel_aggregated
    >>> scores = {'acc': 0.9494, 'sens': 0.7450, 'spec': 0.9793}
    >>> k = 4
    >>> results = check_drive_vessel_aggregated(scores=scores,
                                                eps=10**(-k),
                                                imageset='test',
                                                annotator=1,
                                                verbosity=0)
    >>> results['inconsistency']
    # {'inconsistency_fov_mos': False,
    #  'inconsistency_fov_som': False,
    #  'inconsistency_all_mos': True,
    #  'inconsistency_all_som': True}

The results here show that the reported scores could not be the result of any aggregation of any evaluation methodologies.

STARE dataset
^^^^^^^^^^^^^

The STARE [STARE]_ dataset consists of 20 images for the segmentation of retinal vessels. Two sets of ground truth images are provided as binary masks of the same size as the images. The masks are provided in the PNG format. Usually, the evaluation of segmentation techniques on the STARE dataset is carried out by selecting one subset of the images for training and another for testing.

The consistency testing functions provide tests for individual images and scores aggregated over multiple images, as well.

In the first example, we test the consistency of scores provided for the image 'im0235' against the annotations of the annotator 'ah':

.. code-block:: Python

    >>> from mlscorecheck.bundles.retina import check_stare_vessel_image
    >>> img_identifier = 'im0235'
    >>> scores = {'acc': 0.4699, 'npv': 0.8993, 'f1p': 0.134}
    >>> results = check_stare_vessel_image(image_identifier=img_identifier,
                                            annotator='ah',
                                            scores=scores,
                                            eps=1e-4)
    >>> results['inconsistency']
    # False

In the next example, we illustrate the consistency testing of performance scores aggregated on all figures. As the results show, the reported scores are inconsistent with the assumption of using the SoM aggregation, but there is no evidence for inconsistency with the MoS aggregation.

.. code-block:: Python

    >>> from mlscorecheck.bundles.retina import check_stare_vessel_aggregated
    >>> scores = {'acc': 0.4964, 'sens': 0.5793, 'spec': 0.4871, 'bacc': 0.5332}
    >>> results = check_stare_vessel_aggregated(imageset='all',
                                                annotator='ah',
                                                scores=scores,
                                                eps=1e-4,
                                                verbosity=0)
    >>> results['inconsistency']
    # {'inconsistency_mos': False, 'inconsistency_som': True}


HRF dataset
^^^^^^^^^^^

The HRF [HRF]_ dataset consists of 45 images for the segmentation of retinal vessels. The ground truth is provided as binary masks of the same size as the images. The images are provided with the Field-of-View (FoV) masks, which are also binary masks of the same size as the images.

In the first example, we illustrate the consistency testing of scores reported for one single image ('13_h'):

.. code-block:: Python

    >>> from mlscorecheck.bundles.retina import check_hrf_vessel_image
    >>> scores = {'acc': 0.5562, 'sens': 0.5049, 'spec': 0.5621}
    >>> identifier = '13_h'
    >>> k = 4
    >>> results = check_hrf_vessel_image(scores=scores,
                                            eps=10**(-k),
                                            image_identifier=identifier)
    >>> results['inconsistency']
    # {'inconsistency_fov': False, 'inconsistency_all': True}

The results show that the scores are inconsistent with the assumption of using all pixels for evaluation, but there is no evidence for inconsistency with the assumption of using the FoV mask.

In the next example, we illustrate the consistency testing of scores aggregated over all images:

.. code-block:: Python

    >>> from mlscorecheck.bundles.retina import check_hrf_vessel_aggregated
        >>> scores = {'acc': 0.4841, 'sens': 0.5665, 'spec': 0.475}
        >>> k = 4
        >>> results = check_hrf_vessel_aggregated(scores=scores,
                                                    eps=10**(-k),
                                                    imageset='all',
                                                    verbosity=0)
        >>> results['inconsistency']
        # {'inconsistency_fov_mos': False,
        # 'inconsistency_fov_som': True,
        # 'inconsistency_all_mos': False,
        # 'inconsistency_all_som': True}

The results show that the scores are inconsistent with any assumptions on the region of evaluation using the SoM aggregation, however, the MoS aggregation could have yielded these scores with using the FoV mask or using all pixels for evaluation, as well.

CHASE_DB1 dataset
^^^^^^^^^^^^^^^^^

The CHASE_DB1 [CHASE_DB1]_ datasets consists of 28 images for the segmentation of retinal vessels. The ground truth is provided as binary masks of the same size as the images.

In the first example, we illustrate the consistency testing of scores reported for one single image ('11R') against the annotations 'manual1':

.. code-block:: Python

    >>> from mlscorecheck.bundles.retina import check_chasedb1_vessel_image
    >>> img_identifier = '11R'
    >>> scores = {'acc': 0.4457, 'sens': 0.0051, 'spec': 0.4706}
    >>> results = check_chasedb1_vessel_image(image_identifier=img_identifier,
                                            annotator='manual1',
                                            scores=scores,
                                            eps=1e-4)
    >>> results['inconsistency']
    # False

The results show that the scores are not found to be inconsistent with the experiment.

The next example illustrates the consistency testing of scores aggregated over all images:

.. code-block:: Python

    >>> from mlscorecheck.bundles.retina import check_chasedb1_vessel_aggregated
    >>> scores = {'acc': 0.5063, 'sens': 0.4147, 'spec': 0.5126}
    >>> k = 4
    >>> results = check_chasedb1_vessel_aggregated(imageset='all',
                                                annotator='manual1',
                                                scores=scores,
                                                eps=1e-4,
                                                verbosity=0)
    >>> results['inconsistency']
    # {'inconsistency_mos': False, 'inconsistency_som': True}

As the results show, the scores are inconsistent with the assumption of using the SoM aggregation, but there is no evidence for inconsistency with the MoS aggregation.

DIARETDB0 dataset
^^^^^^^^^^^^^^^^^

The DIARETDB0 [DIARETDB0]_ dataset consists of 130 images for the detection of retinal lesions. The images are separated to train and test sets in 9 different batches, and the images are labelled by 5 labels:

* neovascularisation,
* hardexudates,
* softexudates,
* hemorrhages,
* redsmalldots.

The consistency tests check the consistency of assigning a label (or a set of labels) to images correctly in a binary classification scenario.

In the following example we illustrate the evaluation of scores aggregated over the test images of all batches, assuming that the task is to assign the label 'hardexudates' to images correctly (i.e. the positive class is 'hardexudates' and the negative class is 'not hardexudates'):

.. code-block:: Python

    >>> from mlscorecheck.bundles.retina import check_diaretdb0_class
    >>> scores = {'acc': 0.4271, 'sens': 0.406, 'spec': 0.4765}
    >>> results = check_diaretdb0_class(subset='test',
                                        batch='all',
                                        class_name='hardexudates',
                                        scores=scores,
                                        eps=1e-4)
    >>> results['inconsistency']
    # {'inconsistency_som': True, 'inconsistency_mos': False}

As the results show, the scores are consistent with the assumption of using the SoM aggregation, but the scores could not have been obtained by the MoS aggregation.

DIARETDB1 dataset
^^^^^^^^^^^^^^^^^

The DIARETDB1 [DIARETDB1]_ dataset consists of 89 images for the detection and segmentation of retinal lesions. The ground truth segmentations of 4 lesions (hardexudates, softexudates, hemorrhages, redsmalldots) are provided as a soft-maps unifying the manual annotations of multiple experts. The ground truth images have to be thresholded at a particular level of confidence to get a hard segmentation, which is used for evaluation. The authors suggest the use of the confidence threshold 0.75. A well defined train and test set of images is specified.

The consistency testing supports the both the testing of image labeling (recognizing whether a particular lesion is present) and the testing of the pixel level segmentation, both at the image level and in aggregations.

In the first example, we illustrate the consistency testing of image labeling, the reported scores are supposed to represent the performance of labeling the images wether hard- or soft-exudates (the positive class) are present or not present (the negative class). According to the suggestion of the authors of the dataset, the ground truth images are thresholded at the confidence level 0.75 (pixel values thresholded at 0.75*255). The consistency testing is performed on the test images of the dataset.

.. code-block:: Python

    >>> from mlscorecheck.bundles.retina import check_diaretdb1_class
    >>> scores = {'acc': 0.3115, 'sens': 1.0, 'spec': 0.0455, 'f1p': 0.4474}
    >>> results = check_diaretdb1_class(subset='test',
                            class_name=['hardexudates', 'softexudates'],
                            confidence=0.75,
                            scores=scores,
                            eps=1e-4)
    >>> results['inconsistency']
    # False

As the test results shows, inconsistencies were not identified, the scores could be the outcome of the experiment.

In the next example, we test if the reported scores could be yielded from the segmentation all exudates in one image ('005'), testing both assumptions of using the FoV mask and using all pixels for evaluation:

.. code-block:: Python

    >>> from mlscorecheck.bundles.retina import check_diaretdb1_segmentation_image
    >>> scores = {'acc': 0.5753, 'sens': 0.0503, 'spec': 0.6187, 'f1p': 0.0178}
    >>> results = check_diaretdb1_segmentation_image(image_identifier='005',
                            class_name=['hardexudates', 'softexudates'],
                            confidence=0.75,
                            scores=scores,
                            eps=1e-4)
    >>> results['inconsistency']
    # {'inconsistency_fov': True, 'inconsistency_all': False}

As the results show, the scores are compatible with the use of all pixels for evaluation, but inconsistent with the assumption of using the pixels covered by the FoV mask.

In the last example, we illustrate the consistency testing of scores aggregated over all images of the test set, assuming that the task is to segment all exudates in the images:

.. code-block:: Python

    >>> from mlscorecheck.bundles.retina import check_diaretdb1_segmentation_aggregated
    >>> scores = {'acc': 0.7143, 'sens': 0.3775, 'spec': 0.7244}
    >>> results = check_diaretdb1_segmentation_aggregated(subset='test',
                            class_name='hardexudates',
                            confidence=0.5,
                            only_valid=True,
                            scores=scores,
                            eps=1e-4)
    >>> results['inconsistency']
    # {'inconsistency_fov_som': True,
    # 'inconsistency_all_som': True,
    # 'inconsistency_fov_mos': False,
    # 'inconsistency_all_mos': False}

As the results show, the scores are compatible with the use of SoM aggregation (with both assumptions on the region of evaluation), but the scores are inconsistent with the MoS aggregation (with both assumptions on the region of evaluation).

DRISHTI_GS dataset
^^^^^^^^^^^^^^^^^^

The DRISHTI_GS [DRISHTI_GS]_ dataset consists of 50 training and 51 test images for the segmentation of the optic disk and optic cup in retinal images. The ground truth segmentations are provided as a soft-map, which needs to be thresholded at a certain confidence level to gain hard segmentation labels.

The consistenty tests support the testing of image level and aggregated segmentation results at certain confidence thresholds.

In the first example, we illustrate the consistency testing of image level segmentation results, assuming that the task is to segment the optic disk in the image ('053'):

.. code-block:: Python

    >>> from mlscorecheck.bundles.retina import check_drishti_gs_segmentation_image
    >>> scores = {'acc': 0.5966, 'sens': 0.3, 'spec': 0.6067, 'f1p': 0.0468}
    >>> results = check_drishti_gs_segmentation_image(image_identifier='053',
                                confidence=0.75,
                                target='OD',
                                scores=scores,
                                eps=1e-4)
    >>> results['inconsistency']
    # False

The results show that the scores are consistent with the experiment.

In the next example, we illustrate the consistency testing of scores aggregated over all images of the test set, assuming that the task is to segment the optic disk in the images:

.. code-block:: Python

    >>> from mlscorecheck.bundles.retina import check_drishti_gs_segmentation_aggregated
    >>> scores = {'acc': 0.4767, 'sens': 0.4845, 'spec': 0.4765, 'f1p': 0.0512}
    >>> results = check_drishti_gs_segmentation_aggregated(subset='test',
                                confidence=0.75,
                                target='OD',
                                scores=scores,
                                eps=1e-4)
    >>> results['inconsistency']
    # {'inconsistency_som': False, 'inconsistency_mos': False}

As the results show, the scores are consistent with both assumptions on the mode of aggregation (SoM and MoS).
