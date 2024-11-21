import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

import common_datasets.binary_classification as binclas

datasets = binclas.get_filtered_data_loaders(
    n_col_bounds=(0, 50), 
    n_bounds=(100, 10000), 
    n_minority_bounds=(20, 1000), 
    n_from_phenotypes=1, 
    imbalance_ratio_bounds=(0.2, 20.0)
)

datasets = [dataset for dataset in datasets if not dataset()['name'].startswith('led')]

def generate_random_classifier(random_state, p=None, n=None):

    if p is not None and n is not None:
        min_class = min(p, n)
        n_class = p + n
    else:
        min_class = None
        n_class = None

    mode = random_state.randint(5)
    if mode == 0:
        classifier = RandomForestClassifier
        if min_class is None:
            params = {'max_depth': random_state.randint(2, 10),
                    'random_state': 5}
        else:
            frac = random_state.choice([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3])
            params = {'max_depth': max(int(frac * n_class), 1),
                    'random_state': 5}
    if mode == 1:
        classifier = DecisionTreeClassifier
        if min_class is None:
            params = {'max_depth': random_state.randint(2, 10),
                    'random_state': 5}
        else:
            frac = random_state.choice([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3])
            params = {'max_depth': max(int(frac * n_class), 1),
                    'random_state': 5}
    if mode == 2:
        classifier = SVC
        params = {'probability': True, 'C': random_state.rand()/2 + 0.001, 'tol': 1e-4}
    if mode == 3:
        classifier = KNeighborsClassifier
        params = {'n_neighbors': random_state.randint(2, int(np.sqrt(n_class)))}
    if mode == 4:
        classifier = XGBClassifier
        params = {'random_state': 5, 'max_depth': random_state.randint(2, max(3, int(np.log(n_class))))}
    
    return (classifier, params)
