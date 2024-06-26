.. raw:: html
    :file: ga4.html

The main interface
******************

Consistency testing (``check``)
===============================

The test functions implemented in the ``mlscorecheck.check`` module.

Binary classification
---------------------

.. autofunction:: mlscorecheck.check.binary.check_1_testset_no_kfold
.. autofunction:: mlscorecheck.check.binary.check_1_dataset_kfold_som
.. autofunction:: mlscorecheck.check.binary.check_1_dataset_known_folds_mos
.. autofunction:: mlscorecheck.check.binary.check_1_dataset_unknown_folds_mos
.. autofunction:: mlscorecheck.check.binary.check_n_testsets_mos_no_kfold
.. autofunction:: mlscorecheck.check.binary.check_n_testsets_som_no_kfold
.. autofunction:: mlscorecheck.check.binary.check_n_datasets_som_kfold_som
.. autofunction:: mlscorecheck.check.binary.check_n_datasets_mos_kfold_som
.. autofunction:: mlscorecheck.check.binary.check_n_datasets_mos_known_folds_mos
.. autofunction:: mlscorecheck.check.binary.check_n_datasets_mos_unknown_folds_mos

Multiclass classification
-------------------------

.. autofunction:: mlscorecheck.check.multiclass.check_1_testset_no_kfold_macro
.. autofunction:: mlscorecheck.check.multiclass.check_1_testset_no_kfold_micro
.. autofunction:: mlscorecheck.check.multiclass.check_1_dataset_known_folds_mos_macro
.. autofunction:: mlscorecheck.check.multiclass.check_1_dataset_known_folds_mos_micro
.. autofunction:: mlscorecheck.check.multiclass.check_1_dataset_known_folds_som_macro
.. autofunction:: mlscorecheck.check.multiclass.check_1_dataset_known_folds_som_micro

Regression
----------

.. autofunction:: mlscorecheck.check.regression.check_1_testset_no_kfold

Test bundles (``bundles``)
==========================

The test bundles dedicated to specific problems in the ``mlscorecheck.bundles`` module.

Retina Image Processing
-----------------------

The test functions dedicated to retina image processing problems.

DRIVE
~~~~~

.. autofunction:: mlscorecheck.check.bundles.retina.check_drive_vessel_image
.. autofunction:: mlscorecheck.check.bundles.retina.check_drive_vessel_image_assumption
.. autofunction:: mlscorecheck.check.bundles.retina.check_drive_vessel_aggregated
.. autofunction:: mlscorecheck.check.bundles.retina.check_drive_vessel_aggregated_mos_assumption
.. autofunction:: mlscorecheck.check.bundles.retina.check_drive_vessel_aggregated_som_assumption

STARE
~~~~~

.. autofunction:: mlscorecheck.check.bundles.retina.check_stare_vessel_image
.. autofunction:: mlscorecheck.check.bundles.retina.check_stare_vessel_aggregated
.. autofunction:: mlscorecheck.check.bundles.retina.check_stare_vessel_aggregated_mos
.. autofunction:: mlscorecheck.check.bundles.retina.check_stare_vessel_aggregated_som

HRF
~~~

.. autofunction:: mlscorecheck.check.bundles.retina.check_hrf_vessel_image
.. autofunction:: mlscorecheck.check.bundles.retina.check_hrf_vessel_image_assumption
.. autofunction:: mlscorecheck.check.bundles.retina.check_hrf_vessel_aggregated
.. autofunction:: mlscorecheck.check.bundles.retina.check_hrf_vessel_aggregated_mos_assumption
.. autofunction:: mlscorecheck.check.bundles.retina.check_hrf_vessel_aggregated_som_assumption

CHASE_DB1
~~~~~~~~~

.. autofunction:: mlscorecheck.check.bundles.retina.check_chasedb1_vessel_image
.. autofunction:: mlscorecheck.check.bundles.retina.check_chasedb1_vessel_aggregated
.. autofunction:: mlscorecheck.check.bundles.retina.check_chasedb1_vessel_aggregated_mos
.. autofunction:: mlscorecheck.check.bundles.retina.check_chasedb1_vessel_aggregated_som

DIARETDB0
~~~~~~~~~

.. autofunction:: mlscorecheck.check.bundles.retina.check_diaretdb0_class

DIARETDB1
~~~~~~~~~

.. autofunction:: mlscorecheck.check.bundles.retina.check_diaretdb1_class
.. autofunction:: mlscorecheck.check.bundles.retina.check_diaretdb1_segmentation_image
.. autofunction:: mlscorecheck.check.bundles.retina.check_diaretdb1_segmentation_aggregated

DRISHTI_GS
~~~~~~~~~~

.. autofunction:: mlscorecheck.check.bundles.retina.check_drishti_gs_segmentation_image
.. autofunction:: mlscorecheck.check.bundles.retina.check_drishti_gs_segmentation_aggregated

Preterm delivery prediction by EHG signals
------------------------------------------

The test bundle dedicated to the testing of electrohsyterogram data.

.. autofunction:: mlscorecheck.check.bundles.ehg.check_tpehg

Skin lesion classification
--------------------------

The test bundle dedicated to the testing of skin lesion classification.

ISIC2016
~~~~~~~~

.. autofunction:: mlscorecheck.check.bundles.skinlesion.check_isic2016

ISIC2017
~~~~~~~~

.. autofunction:: mlscorecheck.check.bundles.skinlesion.check_isic2017

Experiments (``experiments``)
=============================

The predefined dataset and experiment statistics to look up are stored in the ``mlscorecheck.experiments`` module.

.. autofunction:: mlscorecheck.experiments.load_ml_datasets
.. autofunction:: mlscorecheck.experiments.lookup_dataset
.. autofunction:: mlscorecheck.experiments.load_drive

The core modules
****************

Score functions (``scores``)
============================

.. autofunction:: mlscorecheck.scores.accuracy
.. autofunction:: mlscorecheck.scores.error_rate
.. autofunction:: mlscorecheck.scores.sensitivity
.. autofunction:: mlscorecheck.scores.false_negative_rate
.. autofunction:: mlscorecheck.scores.false_positive_rate
.. autofunction:: mlscorecheck.scores.specificity
.. autofunction:: mlscorecheck.scores.positive_predictive_value
.. autofunction:: mlscorecheck.scores.false_discovery_rate
.. autofunction:: mlscorecheck.scores.false_omission_rate
.. autofunction:: mlscorecheck.scores.negative_predictive_value
.. autofunction:: mlscorecheck.scores.f_beta_positive
.. autofunction:: mlscorecheck.scores.f_beta_negative
.. autofunction:: mlscorecheck.scores.f1_positive
.. autofunction:: mlscorecheck.scores.f1_negative
.. autofunction:: mlscorecheck.scores.unified_performance_measure
.. autofunction:: mlscorecheck.scores.geometric_mean
.. autofunction:: mlscorecheck.scores.fowlkes_mallows_index
.. autofunction:: mlscorecheck.scores.markedness
.. autofunction:: mlscorecheck.scores.positive_likelihood_ratio
.. autofunction:: mlscorecheck.scores.negative_likelihood_ratio
.. autofunction:: mlscorecheck.scores.matthews_correlation_coefficient
.. autofunction:: mlscorecheck.scores.bookmaker_informedness
.. autofunction:: mlscorecheck.scores.prevalence_threshold
.. autofunction:: mlscorecheck.scores.diagnostic_odds_ratio
.. autofunction:: mlscorecheck.scores.jaccard_index
.. autofunction:: mlscorecheck.scores.balanced_accuracy
.. autofunction:: mlscorecheck.scores.cohens_kappa

.. autofunction:: mlscorecheck.scores.accuracy_standardized
.. autofunction:: mlscorecheck.scores.error_rate_standardized
.. autofunction:: mlscorecheck.scores.sensitivity_standardized
.. autofunction:: mlscorecheck.scores.false_negative_rate_standardized
.. autofunction:: mlscorecheck.scores.false_positive_rate_standardized
.. autofunction:: mlscorecheck.scores.specificity_standardized
.. autofunction:: mlscorecheck.scores.positive_predictive_value_standardized
.. autofunction:: mlscorecheck.scores.false_discovery_rate_standardized
.. autofunction:: mlscorecheck.scores.false_omission_rate_standardized
.. autofunction:: mlscorecheck.scores.negative_predictive_value_standardized
.. autofunction:: mlscorecheck.scores.f_beta_positive_standardized
.. autofunction:: mlscorecheck.scores.f_beta_negative_standardized
.. autofunction:: mlscorecheck.scores.f1_positive_standardized
.. autofunction:: mlscorecheck.scores.f1_negative_standardized
.. autofunction:: mlscorecheck.scores.unified_performance_measure_standardized
.. autofunction:: mlscorecheck.scores.geometric_mean_standardized
.. autofunction:: mlscorecheck.scores.fowlkes_mallows_index_standardized
.. autofunction:: mlscorecheck.scores.markedness_standardized
.. autofunction:: mlscorecheck.scores.positive_likelihood_ratio_standardized
.. autofunction:: mlscorecheck.scores.negative_likelihood_ratio_standardized
.. autofunction:: mlscorecheck.scores.matthews_correlation_coefficient_standardized
.. autofunction:: mlscorecheck.scores.bookmaker_informedness_standardized
.. autofunction:: mlscorecheck.scores.prevalence_threshold_standardized
.. autofunction:: mlscorecheck.scores.diagnostic_odds_ratio_standardized
.. autofunction:: mlscorecheck.scores.jaccard_index_standardized
.. autofunction:: mlscorecheck.scores.balanced_accuracy_standardized
.. autofunction:: mlscorecheck.scores.cohens_kappa_standardized


Testing logic for individual scores (``individual``)
====================================================

The main, low level interface function of the module is ``check_scores_tptn_pairs``.

.. autofunction:: mlscorecheck.individual.check_scores_tptn_pairs

Testing logic for aggregated scores (``aggregated``)
====================================================

The main, low level interface function of the module is ``check_aggregated_scores``.

.. autofunction:: mlscorecheck.aggregated.check_aggregated_scores
.. autoclass:: mlscorecheck.aggregated.Dataset
    :members:
.. autoclass:: mlscorecheck.aggregated.Folding
    :members:
.. autoclass:: mlscorecheck.aggregated.Fold
    :members:
.. autoclass:: mlscorecheck.aggregated.Evaluation
    :members:
.. autoclass:: mlscorecheck.aggregated.Experiment
    :members:
