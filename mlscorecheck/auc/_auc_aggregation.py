"""
This module implements the aggregated AUC related functionalities
"""

import numpy as np

from cvxopt import matrix
from cvxopt.solvers import qp, cp, cpl

from ._auc_single import (
    translate,
    prepare_intervals_for_auc_estimation
)

__all__ = [
    "auc_from_sens_spec_kfold",
    "generate_average",
    "generate_kfold_sens_spec_fix_problem",
    "R"
]


def generate_average(avg_value, n_items, lower_bound=None, random_state=None):
    random_state = (
        np.random.RandomState(random_state)
        if not isinstance(random_state, np.random.RandomState)
        else random_state
    )

    if lower_bound is not None:
        if avg_value < lower_bound:
            raise ValueError("The average value cannot be less than the lower bound")

    values = np.repeat(avg_value, n_items)

    indices = list(range(n_items))

    for _ in range(n_items * 10):
        a, b = random_state.choice(indices, 2, replace=False)
        if random_state.randint(2) == 0:
            dist = min(values[a], 1 - values[a], values[b], 1 - values[b])
            d = random_state.random() * dist

            if lower_bound is not None and values[b] - d < lower_bound:
                d = values[b] - lower_bound
            values[a] += d
            values[b] -= d
        else:
            mean = (values[a] + values[b]) / 2
            values[a] = (values[a] + mean) / 2
            values[b] = (values[b] + mean) / 2

    return values.astype(float)


def generate_kfold_sens_spec_fix_problem(
    *, sens, spec, k, sens_lower_bound=None, spec_lower_bound=None, random_state=None
):
    return {
        "sens": generate_average(sens, k, sens_lower_bound, random_state),
        "spec": generate_average(spec, k, spec_lower_bound, random_state),
    }


def R(x: float, k: int) -> list:
    result = []
    x = x * k
    while x >= 1:
        result.append(1)
        x = x - 1
    result.append(x)
    while len(result) < k:
        result.append(0)

    return result

def auc_from_sens_spec_kfold(
    *,
    scores: dict,
    eps: float,
    p: int,
    n: int,
    lower: str = "min",
    upper: str = "max",
    k: int = None,
    raise_error: bool = False
) -> tuple:
    """
    This module applies the estimation scheme A to estimate AUC from scores
    in k-fold evaluation

    Args:
        scores (dict): the reported scores average scores
        eps (float): the numerical uncertainty
        p (int): the number of positive samples
        n (int): the number of negative samples
        lower (str): ('min'/'cmin') - the type of estimation for the lower bound
        upper (str): ('max'/'amax') - the type of estimation for the upper bound
        k (int): the number of folds (if any)

    Returns:
        tuple(float, float): the interval for the average AUC
    """

    scores = translate(scores)

    if ("sens" in scores) + ("spec" in scores) + ("acc" in scores) < 2:
        raise ValueError("Not enough scores specified for the estimation")

    if p is None or n is None:
        raise ValueError("For k-fold estimation p and n are needed")
    if p % k != 0 or n % k != 0:
        raise ValueError("For k-fold, p and n must be divisible by k")

    intervals = prepare_intervals_for_auc_estimation(scores, eps, p, n)

    if lower == "min":
        RL_avg_sens = R(intervals["sens"][0], k)
        RL_avg_spec = R(intervals["spec"][0], k)

        lower0 = np.mean([a * b for a, b in zip(RL_avg_sens, RL_avg_spec[::-1])])
    elif lower == "cmin":
        if intervals["sens"][0] < 1 - intervals["spec"][0]:
            raise ValueError(
                'sens >= 1 - spec does not hold for "\
                            "the corrected minimum curve'
            )
        lower0 = 0.5 + (1 - intervals["spec"][0] - intervals["sens"][0]) ** 2 / 2.0
    else:
        raise ValueError("Unsupported lower bound")

    if upper == "max":
        RU_avg_sens = R(intervals["sens"][1], k)
        RU_avg_spec = R(intervals["spec"][1], k)

        upper0 = 1 - np.mean(
            [(1 - a) * (1 - b) for a, b in zip(RU_avg_sens, RU_avg_spec[::-1])]
        )
    elif upper == "amax":
        if not intervals["acc"][0] >= max(p, n) / (p + n):
            raise ValueError("accuracy too small")

        upper0 = 1 - ((1 - intervals["acc"][1]) * (p + n)) ** 2 / (2 * n * p)
    else:
        raise ValueError("Unsupported upper bound")

    return (float(lower0), float(upper0))


def acc_from_auc_kfold_wrapper(
    *, 
    scores: dict, 
    eps: float, 
    ps: int, 
    ns: int, 
    upper: str = "max"
):
    try:
        return acc_from_auc_kfold(
            scores=scores,
            eps=eps,
            ps=ps,
            ns=ns,
            upper=upper,
            raise_errors=True
        )
    except:
        return None
    

def acc_from_auc_kfold(
    *, 
    scores: dict, 
    eps: float, 
    ps: int, 
    ns: int, 
    upper: str = "max",
    raise_errors: bool = False
) -> tuple:
    """
    This module applies the estimation scheme A to estimate AUC from scores

    Args:
        scores (dict): the reported scores
        eps (float): the numerical uncertainty
        p (int): the number of positive samples
        n (int): the number of negative samples
        upper (str): 'max'/'cmax' - the type of upper bound

    Returns:
        tuple(float, float): the interval for the maximum accuracy
    """

    if not raise_errors:
        return acc_from_auc_kfold_wrapper(
            scores=scores,
            eps=eps,
            ps=ps,
            ns=ns,
            upper=upper
        )

    scores = translate(scores)

    auc = (max(scores["auc"] - eps, 0), min(scores["auc"] + eps, 1))

    #if np.sqrt((1 - auc[0])*2*p*n) > min(p, n):
    if auc[0] < 1 - min(p, n)/(2*max(p, n)):
        lower = 1 - (np.sqrt(2 * p * n - 2 * auc[0] * p * n)) / (p + n)
    else:
        lower = max(p, n)/(p + n)
        #raise ValueError("AUC too small")
    
    if upper == "max":
        upper = (auc[1] * min(p, n) + max(p, n)) / (p + n)
    else:
        upper = (max(p, n) + min(p, n) * np.sqrt(2 * (auc[1] - 0.5))) / (p + n)

    return (float(lower), float(upper))


def auc_amax_kfold(ps, ns, accs):
    return np.mean(1 - (1 - accs)**2*(ps + ns)**2/(2*ps*ns))

def auc_amax_kfold_solve(ps, ns, avg_acc):
    k = ps.shape[0]
    q = (ps + ns)**2/(2*ps*ns)/k*2
    Q = np.diag(q)
    A = np.repeat(1.0/k, k).reshape(-1, 1).T
    b = np.array([avg_acc])
    G = np.vstack([np.eye(k), -np.eye(k)]).astype(float)
    lower = np.array([max(p, n)/(p + n) for p, n in zip(ps, ns)])
    h = np.hstack([np.repeat(1.0, k), -lower])
    
    Q = matrix(Q)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)

    res = qp(P=Q, q=q, G=G, h=h, A=A, b=b)

    print(res['x'])

    return auc_amax_kfold(ps, ns, np.array(res['x']))

class F_acc_cmax:
    def __init__(self, ps, ns, avg_auc):
        self.ps = ps
        self.ns = ns
        self.avg_auc = avg_auc
        self.k = len(ps)
        self.maxs = np.array([max(p, n) for p, n in zip(ps, ns)])
        self.mins = np.array([min(p, n) for p, n in zip(ps, ns)])
        self.weights = self.mins / (ps + ns)

    def __call__(self, x=None, z=None):
        if x is None and z is None:
            return (0, matrix(1.0, (self.k, 1)))
        
        if x is not None:
            f = matrix(-np.sum(np.sqrt(x)*self.weights.reshape(-1, 1)))
            Df = matrix(-0.5/np.sqrt(x)*self.weights.reshape(-1, 1)).T
        
        if z is None:
            return (f, Df)
        
        hess = np.diag(0.5**2*np.array(x)[:, 0]**(-3/2)*self.weights)
        
        hess = matrix(z[0]*hess)

        return (f, Df, hess)


def acc_cmax_kfold(ps, ns, aucs):
    maxs = np.array([max(p, n) for p, n in zip(ps, ns)])
    mins = np.array([min(p, n) for p, n in zip(ps, ns)])

    return np.mean((maxs + mins * np.sqrt(2*aucs - 1)) / (ps + ns))

def acc_cmax_kfold_solve(ps, ns, avg_auc):
    F = F_acc_cmax(ps, ns, avg_auc)

    k = ps.shape[0]
    q = (ps + ns)**2/(2*ps*ns)/k*2
    Q = np.diag(q)
    A = np.repeat(1.0/k, k).reshape(-1, 1).T
    b = np.array([2*avg_auc - 1])
    G = np.vstack([np.eye(k), -np.eye(k)]).astype(float)
    h = np.hstack([np.repeat(1.0, k), -np.repeat(0, k)])
    
    Q = matrix(Q)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)

    res = cp(F, G, h, A=A, b=b)

    aucs = (np.array(res['x']) + 1)/2

    print(aucs)

    return acc_cmax_kfold(ps, ns, aucs)

class F_acc_amin:
    def __init__(self, ps, ns, avg_auc):
        self.ps = ps
        self.ns = ns
        self.avg_auc = avg_auc
        self.k = len(ps)
        self.maxs = np.array([max(p, n) for p, n in zip(ps, ns)])
        self.mins = np.array([min(p, n) for p, n in zip(ps, ns)])
        self.weights = np.sqrt(ps*ns)/(ps + ns)

    def __call__(self, x=None, z=None):
        if x is None and z is None:
            return (0, matrix(1.0, (self.k, 1)))
        
        if x is not None:
            f = matrix(-np.sum(np.sqrt(x)*self.weights.reshape(-1, 1)))
            Df = matrix(-0.5/np.sqrt(x)*self.weights.reshape(-1, 1)).T
        
        if z is None:
            return (f, Df)
        
        hess = np.diag(0.5**2*np.array(x)[:, 0]**(-3/2)*self.weights)
        
        hess = matrix(z[0]*hess)

        return (f, Df, hess)

def acc_amin_kfold(ps, ns, aucs):
    maxs = np.array([max(p, n) for p, n in zip(ps, ns)])
    mins = np.array([min(p, n) for p, n in zip(ps, ns)])

    return np.mean(1 - np.sqrt(2*(1 - aucs)*ps*ns)/(ps + ns))

def acc_amin_kfold_solve(ps, ns, avg_auc):
    F = F_acc_cmax(ps, ns, avg_auc)

    lower_bounds = 2*(1 - 1.0 - F.mins/(2*F.maxs))

    k = ps.shape[0]
    q = (ps + ns)**2/(2*ps*ns)/k*2
    Q = np.diag(q)
    A = np.repeat(1.0/k, k).reshape(-1, 1).T
    b = np.array([2*(1 - avg_auc)])
    G = np.vstack([np.eye(k), -np.eye(k)]).astype(float)
    h = np.hstack([np.repeat(1.0, k), -lower_bounds])
    
    Q = matrix(Q)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)

    res = cp(F, G, h, A=A, b=b)

    aucs = 1 - (np.array(res['x']))/2

    print(aucs)

    return acc_amin_kfold(ps, ns, aucs)