# %% [markdown]
# # All examples used in the README and documentation

# %%
from mlscorecheck.check import check_1_testset_no_kfold_scores

result = check_1_testset_no_kfold_scores(testset={'p': 100, 'n': 200},
                                         scores={'acc': 0.9567, 'sens': 0.8545, 'spec': 0.9734},
                                         eps=1e-4)
result['inconsistency']
# True

# %% [markdown]
# ## 1 testset no kfold

# %%
from mlscorecheck.check import check_1_testset_no_kfold_scores

testset = {'p': 530, 'n': 902}

scores = {'acc': 0.62, 'sens': 0.22, 'spec': 0.86, 'f1p': 0.3, 'fm': 0.32}

result = check_1_testset_no_kfold_scores(testset=testset,
                                        scores=scores,
                                        eps=1e-2)
result['inconsistency']
# False

# %%
scores = {'acc': 0.92, 'sens': 0.22, 'spec': 0.86, 'f1p': 0.3, 'fm': 0.32}

result = check_1_testset_no_kfold_scores(testset=testset,
                                        scores=scores,
                                        eps=1e-2)
result['inconsistency']
# True

# %% [markdown]
# ## 1 dataset with k-fold, mean-of-scores (MoS)

# %%
from mlscorecheck.check import check_1_dataset_known_folds_mos_scores

dataset = {'p': 126, 'n': 131}
folding = {'folds': [{'p': 52, 'n': 94}, {'p': 74, 'n': 37}]}

scores = {'acc': 0.573, 'sens': 0.768, 'bacc': 0.662}

result = check_1_dataset_known_folds_mos_scores(dataset=dataset,
                                                folding=folding,
                                                scores=scores,
                                                eps=1e-3)
result['inconsistency']
# False

# %%
scores = {'acc': 0.573, 'sens': 0.568, 'bacc': 0.662}
result = check_1_dataset_known_folds_mos_scores(dataset=dataset,
                                                folding=folding,
                                                scores=scores,
                                                eps=1e-3)
result['inconsistency']
# True

# %%
dataset = {'dataset_name': 'common_datasets.glass_0_1_6_vs_2'}
folding = {'n_folds': 4, 'n_repeats': 2, 'strategy': 'stratified_sklearn'}

scores = {'acc': 0.9, 'spec': 0.9, 'sens': 0.6, 'bacc': 0.1, 'f1': 0.95}

result = check_1_dataset_known_folds_mos_scores(dataset=dataset,
                                                folding=folding,
                                                fold_score_bounds={'acc': (0.8, 1.0)},
                                                scores=scores,
                                                eps=1e-2,
                                                numerical_tolerance=1e-6)
result['inconsistency']
# True

# %% [markdown]
# ## 1 dataset with kfold ratio-of-means (SoM)

# %%
from mlscorecheck.check import check_1_dataset_som_scores

dataset = {'dataset_name': 'common_datasets.monk-2'}
folding = {'n_folds': 4, 'n_repeats': 3, 'strategy': 'stratified_sklearn'}

scores = {'spec': 0.668, 'npv': 0.744, 'ppv': 0.667,
            'bacc': 0.706, 'f1p': 0.703, 'fm': 0.704}

result = check_1_dataset_som_scores(dataset=dataset,
                                    folding=folding,
                                    scores=scores,
                                    eps=1e-3)
result['inconsistency']
# False

# %%
scores = {'spec': 0.668, 'npv': 0.754, 'ppv': 0.667,
        'bacc': 0.706, 'f1p': 0.703, 'fm': 0.704}

result = check_1_dataset_som_scores(dataset=dataset,
                                    folding=folding,
                                    scores=scores,
                                    eps=1e-3)
result['inconsistency']
# True

# %% [markdown]
# ## N datasets with k-folds, SoM over datasets and SoM over folds

# %%
from mlscorecheck.check import check_n_datasets_som_kfold_som_scores

evaluation0 = {'dataset': {'p': 389, 'n': 630},
                'folding': {'n_folds': 5, 'n_repeats': 2,
                            'strategy': 'stratified_sklearn'}}
evaluation1 = {'dataset': {'dataset_name': 'common_datasets.saheart'},
                'folding': {'n_folds': 5, 'n_repeats': 2,
                            'strategy': 'stratified_sklearn'}}
evaluations = [evaluation0, evaluation1]

scores = {'acc': 0.631, 'sens': 0.341, 'spec': 0.802, 'f1p': 0.406, 'fm': 0.414}

result = check_n_datasets_som_kfold_som_scores(scores=scores,
                                                evaluations=evaluations,
                                                eps=1e-3)
result['inconsistency']
# False

# %%
scores = {'acc': 0.731, 'sens': 0.341, 'spec': 0.802, 'f1p': 0.406, 'fm': 0.414}

result = check_n_datasets_som_kfold_som_scores(scores=scores,
                                                evaluations=evaluations,
                                                eps=1e-3)
result['inconsistency']
# True

# %% [markdown]
# ## N datasets with k-folds, MoS over datasets and SoM over folds

# %%
from mlscorecheck.check import check_n_datasets_mos_kfold_som_scores

evaluation0 = {'dataset': {'p': 39, 'n': 822},
                'folding': {'n_folds': 5, 'n_repeats': 3,
                            'strategy': 'stratified_sklearn'}}
evaluation1 = {'dataset': {'dataset_name': 'common_datasets.winequality-white-3_vs_7'},
                'folding': {'n_folds': 5, 'n_repeats': 3,
                            'strategy': 'stratified_sklearn'}}
evaluations = [evaluation0, evaluation1]

scores = {'acc': 0.312, 'sens': 0.45, 'spec': 0.312, 'bacc': 0.381}

result = check_n_datasets_mos_kfold_som_scores(evaluations=evaluations,
                                                dataset_score_bounds={'acc': (0.0, 0.5)},
                                                eps=1e-4,
                                                scores=scores)
result['inconsistency']
# False

# %%
scores = {'acc': 0.412, 'sens': 0.45, 'spec': 0.312, 'bacc': 0.381}
result = check_n_datasets_mos_kfold_som_scores(evaluations=evaluations,
                                                dataset_score_bounds={'acc': (0.5, 1.0)},
                                                eps=1e-4,
                                                scores=scores)
result['inconsistency']
# True

# %% [markdown]
# ## N datasets with k-folds, MoS over datasets and MoS over folds

# %%
from mlscorecheck.check import check_n_datasets_mos_known_folds_mos_scores

evaluation0 = {'dataset': {'p': 118, 'n': 95},
                'folding': {'folds': [{'p': 22, 'n': 23}, {'p': 96, 'n': 72}]}}
evaluation1 = {'dataset': {'p': 781, 'n': 423},
                'folding': {'folds': [{'p': 300, 'n': 200}, {'p': 481, 'n': 223}]}}
evaluations = [evaluation0, evaluation1]

scores = {'acc': 0.61, 'sens': 0.709, 'spec': 0.461, 'bacc': 0.585}

result = check_n_datasets_mos_known_folds_mos_scores(evaluations=evaluations,
                                                    scores=scores,
                                                    eps=1e-3)
result['inconsistency']
# False

# %%
scores = {'acc': 0.71, 'sens': 0.709, 'spec': 0.461}

result = check_n_datasets_mos_known_folds_mos_scores(evaluations=evaluations,
                                                    scores=scores,
                                                    eps=1e-3)
result['inconsistency']
# True

# %% [markdown]
# ## Not knowing the k-folding scheme

# %%
from mlscorecheck.check import check_1_dataset_unknown_folds_mos_scores

dataset = {'p': 126, 'n': 131}
folding = {'n_folds': 2, 'n_repeats': 1}

scores = {'acc': 0.573, 'sens': 0.768, 'bacc': 0.662}

result = check_1_dataset_unknown_folds_mos_scores(dataset=dataset,
                                                    folding=folding,
                                                    scores=scores,
                                                    eps=1e-3)
result['inconsistency']
# False

# %%
scores = {'acc': 0.573, 'sens': 0.768, 'bacc': 0.862}

result = check_1_dataset_unknown_folds_mos_scores(dataset=dataset,
                                                    folding=folding,
                                                    scores=scores,
                                                    eps=1e-3)
result['inconsistency']
# True

# %%
from mlscorecheck.check import check_n_datasets_mos_unknown_folds_mos_scores

evaluation0 = {'dataset': {'p': 13, 'n': 73},
                'folding': {'n_folds': 4, 'n_repeats': 1}}
evaluation1 = {'dataset': {'p': 7, 'n': 26},
                'folding': {'n_folds': 3, 'n_repeats': 1}}
evaluations = [evaluation0, evaluation1]

scores = {'acc': 0.357, 'sens': 0.323, 'spec': 0.362, 'bacc': 0.343}

result = check_n_datasets_mos_unknown_folds_mos_scores(evaluations=evaluations,
                                                        scores=scores,
                                                        eps=1e-3)
result['inconsistency']
# False

# %%
scores = {'acc': 0.357, 'sens': 0.323, 'spec': 0.362, 'bacc': 0.9}

result = check_n_datasets_mos_unknown_folds_mos_scores(evaluations=evaluations,
                                                        scores=scores,
                                                        eps=1e-3)
result['inconsistency']
# True

# %% [markdown]
# ## Test bundles

# %%
from mlscorecheck.bundles import (drive_image, drive_aggregated)

drive_image(scores={'acc': 0.9478, 'npv': 0.8532, 'f1p': 0.9801, 'ppv': 0.8543},
            eps=1e-4,
            image_set='test',
            identifier='01')
# {'fov_inconsistency': True, 'no_fov_inconsistency': True}

# %%
drive_aggregated(scores={'acc': 0.9478, 'sens': 0.8532, 'spec': 0.9801},
                eps=1e-4,
                image_set='test')
# {'mos_fov_inconsistency': True,
#   'mos_no_fov_inconsistency': True,
#   'som_fov_inconsistency': True,
#   'som_no_fov_inconsistency': True}

# %%
from mlscorecheck.bundles import check_ehg

scores = {'acc': 0.9552, 'sens': 0.9351, 'spec': 0.9713}

results = check_ehg(scores=scores, eps=10**(-4), n_folds=10, n_repeats=1)
results['inconsistency']
# True

# %%



