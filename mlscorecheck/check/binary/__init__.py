"""
This module brings together all the check functionalities
"""

from ._check_1_testset_no_kfold import *

from ._check_1_dataset_kfold_som import *
from ._check_1_dataset_known_folds_mos import *
from ._check_1_dataset_unknown_folds_mos import *

from ._check_n_testsets_mos_no_kfold import *
from ._check_n_testsets_som_no_kfold import *

from ._check_n_datasets_som_kfold_som import *
from ._check_n_datasets_mos_kfold_som import *
from ._check_n_datasets_mos_known_folds_mos import *
from ._check_n_datasets_mos_unknown_folds_mos import *
