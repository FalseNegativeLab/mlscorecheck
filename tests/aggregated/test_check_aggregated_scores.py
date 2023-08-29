"""
This module tests the top level interfaces to the aggregated score checks
"""

from mlscorecheck.aggregated import (check_aggregated_scores,
                                        generate_experiment_specification,
                                        generate_dataset_specification,
                                        Experiment)

def test_check_aggregated_scores_feasible():
    """
    Testing the top level aggregated check function with a feasible problem
    """
    experiment_spec = generate_experiment_specification(random_state=5)
    experiment = Experiment(**experiment_spec)
    sample = experiment.sample(random_state=5)
    scores = sample.calculate_scores(rounding_decimals=3)

    details = check_aggregated_scores(experiment=experiment_spec,
                                            scores=scores,
                                            eps=1e-3)

    assert not details['inconsistency']
    assert details['lp_status'] == 'feasible'
    assert details['lp_configuration_scores_match']
    assert details['lp_configuration_bounds_match']
    assert details['lp_configuration'] is not None

def test_check_aggregated_scores_feasible_custom_solver():
    """
    Testing the top level aggregated check function with a feasible problem
    with custom solver
    """
    experiment_spec = generate_experiment_specification(random_state=5)
    experiment = Experiment(**experiment_spec)
    sample = experiment.sample(random_state=5)
    scores = sample.calculate_scores(rounding_decimals=3)

    details = check_aggregated_scores(experiment=experiment_spec,
                                        scores=scores,
                                        eps=1e-3,
                                        solver_name='dummy')

    assert not details['inconsistency']
    assert details['lp_status'] == 'feasible'
    assert details['lp_configuration_scores_match']
    assert details['lp_configuration_bounds_match']
    assert details['lp_configuration'] is not None

def test_check_aggregated_scores_infeasible():
    """
    Testing the top level aggregated check function with an infeasible problem
    """
    experiment_spec = generate_experiment_specification(random_state=5)
    experiment = Experiment(**experiment_spec)
    sample = experiment.sample(random_state=5)

    scores = {'acc': 0.1, 'sens': 0.1, 'spec': 0.1, 'bacc': 0.4}

    bounds = sample.get_dataset_fold_bounds(score_subset=['acc', 'sens', 'spec', 'bacc'],
                                            feasible=False)

    final = experiment.add_dataset_fold_bounds(bounds)
    final_spec = final.to_dict()

    details = check_aggregated_scores(experiment=final_spec,
                                        scores=scores,
                                        eps=1e-4)

    assert details['inconsistency']
    assert details['lp_status'] == 'infeasible'
    assert details['lp_configuration'] is not None

def test_check_aggregated_scores_timeout():
    """
    Testing the top level aggregated check function with an infeasible problem

    Eventually this test can fail, due to the unpredictability of solvers timing out
    """
    experiment_spec = {'aggregation': 'mor',
                        'datasets': [generate_dataset_specification(aggregation='mor',
                                                                    random_state=idx,
                                                                    max_n_folds=20,
                                                                    max_n_repeats=20)
                                        for idx in range(2)]}

    experiment = Experiment(**experiment_spec)
    sample = experiment.sample(random_state=7)

    scores = sample.calculate_scores(rounding_decimals=7)

    bounds = sample.get_dataset_fold_bounds(['acc', 'sens', 'spec', 'bacc'], True)
    experiment = experiment.add_dataset_fold_bounds(bounds)

    bounds = sample.get_dataset_bounds(['acc', 'sens', 'spec', 'bacc'], True)
    experiment = experiment.add_dataset_bounds(bounds)

    details = check_aggregated_scores(experiment=experiment.to_dict(problem_only=True),
                                        scores=scores,
                                        eps=1e-7,
                                        timeout=0.1,
                                        numerical_tolerance=1e-9)

    assert not details['inconsistency']
    assert details['lp_status'] == 'timeout'
    assert details['lp_configuration'] is not None
