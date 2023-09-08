The main interface
******************

Consistency testing (``check``)
===============================

The test functions implemented in the ``mlscorecheck.check`` module.

.. autofunction:: mlscorecheck.check.check_1_testset_no_kfold_scores
.. autofunction:: mlscorecheck.check.check_1_dataset_kfold_rom_scores
.. autofunction:: mlscorecheck.check.check_1_dataset_kfold_mor_scores
.. autofunction:: mlscorecheck.check.check_n_datasets_rom_kfold_rom_scores
.. autofunction:: mlscorecheck.check.check_n_datasets_mor_kfold_rom_scores
.. autofunction:: mlscorecheck.check.check_n_datasets_mor_kfold_mor_scores


Test bundles (``bundles``)
==========================

The test bundles dedicated to specific problems in the ``mlscorecheck.bundles`` module.

Retinal Vessel Segmentation
---------------------------

.. autofunction:: mlscorecheck.bundles.drive_aggregated
.. autofunction:: mlscorecheck.bundles.drive_aggregated_fov_pixels
.. autofunction:: mlscorecheck.bundles.drive_aggregated_all_pixels
.. autofunction:: mlscorecheck.bundles.drive_image
.. autofunction:: mlscorecheck.bundles.drive_image_fov_pixels
.. autofunction:: mlscorecheck.bundles.drive_image_all_pixels


EHG
---

The test bundle dedicated to the testing of electrohsyterogram data.



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
.. autofunction:: mlscorecheck.scores.f_beta_plus
.. autofunction:: mlscorecheck.scores.f_beta_minus
.. autofunction:: mlscorecheck.scores.f1_plus
.. autofunction:: mlscorecheck.scores.f1_minus
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
.. autofunction:: mlscorecheck.scores.p4

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
.. autofunction:: mlscorecheck.scores.f_beta_plus_standardized
.. autofunction:: mlscorecheck.scores.f_beta_minus_standardized
.. autofunction:: mlscorecheck.scores.f1_plus_standardized
.. autofunction:: mlscorecheck.scores.f1_minus_standardized
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
.. autofunction:: mlscorecheck.scores.p4_standardized


Testing logic for individual scores (``individual``)
====================================================

.. autofunction:: mlscorecheck.individual.check_individual_scores
.. autofunction:: mlscorecheck.individual.check_2v1
.. autofunction:: mlscorecheck.individual.create_intervals
.. autofunction:: mlscorecheck.individual.create_problems_2
.. autofunction:: mlscorecheck.individual.evaluate_1_solution
.. autofunction:: mlscorecheck.individual.check_zero_division
.. autofunction:: mlscorecheck.individual.check_negative_base
.. autofunction:: mlscorecheck.individual.check_empty_interval
.. autofunction:: mlscorecheck.individual.check_intersection
.. autofunction:: mlscorecheck.individual.determine_edge_cases
.. autofunction:: mlscorecheck.individual.resolve_aliases_and_complements
.. autofunction:: mlscorecheck.individual.round_scores
.. autofunction:: mlscorecheck.individual.calculate_scores_for_lp
.. autofunction:: mlscorecheck.individual.calculate_scores
.. autoclass:: mlscorecheck.individual.Expression
    :members:
.. autoclass:: mlscorecheck.individual.Interval
    :members:
.. autoclass:: mlscorecheck.individual.IntervalUnion
    :members:
.. autoclass:: mlscorecheck.individual.Solution
    :members:
.. autoclass:: mlscorecheck.individual.Solutions
    :members:
.. autofunction:: mlscorecheck.individual.load_solutions
.. autofunction:: mlscorecheck.individual.generate_problems
.. autofunction:: mlscorecheck.individual.generate_1_problem
.. autofunction:: mlscorecheck.individual.generate_problem_and_scores

Testing logic for aggregated scores (``aggregated``)
====================================================

.. autofunction:: mlscorecheck.aggregated.check_aggregated_scores
.. autoclass:: mlscorecheck.aggregated.Fold
    :members:
.. autoclass:: mlscorecheck.aggregated.Dataset
    :members:
.. autofunction:: mlscorecheck.aggregated.generate_dataset_specification
.. autofunction:: mlscorecheck.aggregated.create_folds_for_dataset
.. autofunction:: mlscorecheck.aggregated.generate_dataset_and_scores
.. autoclass:: mlscorecheck.aggregated.Experiment
    :members:
.. autofunction:: mlscorecheck.aggregated.generate_dataset_and_scores
.. autofunction:: mlscorecheck.aggregated.generate_dataset_specification
.. autofunction:: mlscorecheck.aggregated.stratified_configurations_sklearn
.. autofunction:: mlscorecheck.aggregated.determine_fold_configurations
.. autofunction:: mlscorecheck.aggregated._create_folds
.. autofunction:: mlscorecheck.aggregated.add_bounds
.. autofunction:: mlscorecheck.aggregated.solve
.. autofunction:: mlscorecheck.aggregated.create_lp_target
.. autofunction:: mlscorecheck.aggregated.random_identifier
.. autofunction:: mlscorecheck.aggregated.check_bounds
.. autofunction:: mlscorecheck.aggregated.compare_scores
.. autofunction:: mlscorecheck.aggregated.create_bounds

Core functions (``core``)
=========================

.. autofunction:: mlscorecheck.core.dict_mean
.. autofunction:: mlscorecheck.core.dict_minmax
.. autofunction:: mlscorecheck.core.load_json
.. autofunction:: mlscorecheck.core.check_uncertainty_and_tolerance
.. autofunction:: mlscorecheck.core.update_uncertainty
.. autofunction:: mlscorecheck.core.init_random_state
.. autofunction:: mlscorecheck.core.round_scores
.. autofunction:: mlscorecheck.core.safe_eval
.. autofunction:: mlscorecheck.core.safe_call


The symbolic toolkit
********************

The symbolic solver (``symbolic``)
==================================

.. automodule:: mlscorecheck.symbolic
    :members:
.. autoclass:: mlscorecheck.symbolic.Algebra
    :members:
.. autoclass:: mlscorecheck.symbolic.SympyAlgebra
    :members:
.. autoclass:: mlscorecheck.symbolic.SageAlgebra
    :members:
.. autoclass:: mlscorecheck.symbolic.Symbols
    :members:
.. autofunction:: mlscorecheck.symbolic.check_importability
.. autofunction:: mlscorecheck.symbolic.get_symbolic_toolkit
.. autoclass:: mlscorecheck.symbolic.ProblemSolver
.. autofunction:: mlscorecheck.symbolic.collect_denominators_and_bases
