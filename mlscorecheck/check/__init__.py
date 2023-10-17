"""
This module brings together all the check functionalities
"""

from ._check_1_testset_no_kfold_scores import *

from ._check_1_dataset_kfold_som_scores import *
from ._check_1_dataset_known_folds_mos_scores import *
from ._check_1_dataset_unknown_folds_mos_acc_score import *
from ._check_1_dataset_unknown_folds_mos_scores import *

from ._check_n_datasets_som_kfold_som_scores import *
from ._check_n_datasets_mos_kfold_som_scores import *
from ._check_n_datasets_mos_known_folds_mos_scores import *
from ._check_n_datasets_mos_unknown_folds_mos_scores import *
