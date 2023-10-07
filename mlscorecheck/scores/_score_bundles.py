"""
This module implements some loader functions for the scores
"""

from ._scores import (accuracy,
                        sensitivity,
                        specificity,
                        negative_predictive_value,
                        positive_predictive_value,
                        balanced_accuracy,
                        f1_positive,
                        f1_negative,
                        f_beta_positive,
                        f_beta_negative,
                        fowlkes_mallows_index,
                        bookmaker_informedness,
                        prevalence_threshold,
                        positive_likelihood_ratio,
                        negative_likelihood_ratio,
                        markedness,
                        diagnostic_odds_ratio,
                        matthews_correlation_coefficient,
                        jaccard_index,
                        cohens_kappa,
                        geometric_mean,
                        unified_performance_measure,
                        error_rate,
                        false_discovery_rate,
                        false_omission_rate,
                        false_negative_rate,
                        false_positive_rate)
from ._scores_standardized import (accuracy_standardized,
                        sensitivity_standardized,
                        specificity_standardized,
                        negative_predictive_value_standardized,
                        positive_predictive_value_standardized,
                        balanced_accuracy_standardized,
                        f1_positive_standardized,
                        f1_negative_standardized,
                        f_beta_positive_standardized,
                        f_beta_negative_standardized,
                        fowlkes_mallows_index_standardized,
                        bookmaker_informedness_standardized,
                        prevalence_threshold_standardized,
                        positive_likelihood_ratio_standardized,
                        negative_likelihood_ratio_standardized,
                        markedness_standardized,
                        diagnostic_odds_ratio_standardized,
                        matthews_correlation_coefficient_standardized,
                        jaccard_index_standardized,
                        cohens_kappa_standardized,
                        geometric_mean_standardized,
                        unified_performance_measure_standardized,
                        error_rate_standardized,
                        false_discovery_rate_standardized,
                        false_omission_rate_standardized,
                        false_negative_rate_standardized,
                        false_positive_rate_standardized)

from ..core import load_json

__all__ = ['score_functions_with_solutions',
            'score_functions_without_complements',
            'score_functions_with_complements',
            'score_functions_standardized_without_complements',
            'score_functions_standardized_with_complements',
            'score_function_aliases',
            'score_function_complements',
            'score_specifications',
            'score_functions_all',
            'score_functions_standardized_all']

# the score specifications
score_specifications = load_json('scores', 'scores.json')['scores']

score_functions_all = {'acc': accuracy,
                'sens': sensitivity,
                'spec': specificity,
                'npv': negative_predictive_value,
                'ppv': positive_predictive_value,
                'bacc': balanced_accuracy,
                'f1p': f1_positive,
                'f1n': f1_negative,
                'fbp': f_beta_positive,
                'fbn': f_beta_negative,
                'fm': fowlkes_mallows_index,
                'bm': bookmaker_informedness,
                'pt': prevalence_threshold,
                'lrp': positive_likelihood_ratio,
                'lrn': negative_likelihood_ratio,
                'mk': markedness,
                'dor': diagnostic_odds_ratio,
                'mcc': matthews_correlation_coefficient,
                'ji': jaccard_index,
                'kappa': cohens_kappa,
                'gm': geometric_mean,
                'upm': unified_performance_measure,
                'err': error_rate,
                'fdr': false_discovery_rate,
                'for_': false_omission_rate,
                'fnr': false_negative_rate,
                'fpr': false_positive_rate}

score_functions_standardized_all = {'acc': accuracy_standardized,
                'sens': sensitivity_standardized,
                'spec': specificity_standardized,
                'npv': negative_predictive_value_standardized,
                'ppv': positive_predictive_value_standardized,
                'bacc': balanced_accuracy_standardized,
                'f1p': f1_positive_standardized,
                'f1n': f1_negative_standardized,
                'fbp': f_beta_positive_standardized,
                'fbn': f_beta_negative_standardized,
                'fm': fowlkes_mallows_index_standardized,
                'bm': bookmaker_informedness_standardized,
                'pt': prevalence_threshold_standardized,
                'lrp': positive_likelihood_ratio_standardized,
                'lrn': negative_likelihood_ratio_standardized,
                'mk': markedness_standardized,
                'dor': diagnostic_odds_ratio_standardized,
                'mcc': matthews_correlation_coefficient_standardized,
                'ji': jaccard_index_standardized,
                'kappa': cohens_kappa_standardized,
                'gm': geometric_mean_standardized,
                'upm': unified_performance_measure_standardized,
                'err': error_rate_standardized,
                'fdr': false_discovery_rate_standardized,
                'for_': false_omission_rate_standardized,
                'fnr': false_negative_rate_standardized,
                'fpr': false_positive_rate_standardized}

# scores with solutions
score_functions_with_solutions = \
    {key: score_functions_all[key]
        for key in ['acc', 'sens', 'spec', 'npv', 'ppv', 'bacc', 'f1p', 'fm',
                    'f1n', 'fbp', 'fbn', 'bm', 'pt', 'lrp', 'lrn', 'mk', 'dor', 'mcc',
                    'ji', 'kappa', 'gm', 'upm']}

# complement functions
score_function_complement_functions = \
    {key: score_functions_all[key]
        for key in ['err', 'fdr', 'for_', 'fnr', 'fpr']}

# all score functions without complements
score_functions_without_complements = \
    {key: score_functions_all[key] for key in
                    ['acc', 'sens', 'spec', 'npv', 'ppv', 'f1p', 'f1n',
                    'fbp', 'fbn', 'bacc', 'fm', 'bm', 'pt', 'lrp', 'lrn',
                    'mk', 'dor', 'mcc', 'ji', 'kappa', 'gm', 'upm']}

# all score functions with complements
score_functions_with_complements = \
    score_functions_without_complements | score_function_complement_functions

# standardized complements
score_function_standardized_complement_functions = \
    {key: score_functions_standardized_all[key]
        for key in ['err', 'fdr', 'for_', 'fnr', 'fpr']}

# all standardized score functions without complements
score_functions_standardized_without_complements = \
    {key: score_functions_standardized_all[key] for key in
                    ['acc', 'sens', 'spec', 'npv', 'ppv', 'f1p', 'f1n',
                    'fbp', 'fbn', 'bacc', 'fm', 'bm', 'pt', 'lrp', 'lrn',
                    'mk', 'dor', 'mcc', 'ji', 'kappa', 'gm', 'upm']}

# all standardized score functions with complements
score_functions_standardized_with_complements = \
    (score_functions_standardized_without_complements
    | score_function_standardized_complement_functions)

# the alias mapping
score_function_aliases = {'tpr': 'sens',
                            'tnr': 'spec',
                            'prec': 'ppv',
                            'rec': 'sens',
                            'f1': 'f1p'}

# the score complement mapping
score_function_complements = {'fdr': 'ppv',
                                'for_': 'npv',
                                'fnr': 'sens',
                                'fpr': 'spec',
                                'err': 'acc'}
