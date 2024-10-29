import numpy as np


def auc_analytic_exponents(row):
    frac = (row['sens']*row['p'] + (1 - row['spec'])*row['n']) / (row['p'] + row['n'])

    print(frac)

    exp_tpr = np.log(row['sens'])/np.log(frac)
    exp_fpr = np.log(1 - row['spec'])/np.log(frac)

    

    return exp_fpr, exp_tpr


def auc_analytic(row, frac = None):
    if frac is None:
        frac = (row['sens']*row['p'] + (1 - row['spec'])*row['n']) / (row['p'] + row['n'])

    print(frac)

    exp_tpr = np.log(row['sens'])/np.log(frac)
    exp_fpr = np.log(1 - row['spec'])/np.log(frac)

    print(exp_tpr, exp_fpr)

    return float(exp_fpr/(exp_fpr + exp_tpr))

    x = np.linspace(0, 1, 100)
    tpr = x**exp_tpr
    fpr = x**exp_fpr

    return float(np.sum((fpr[1:] - fpr[:-1])*(tpr[:-1] + tpr[1:])/2))