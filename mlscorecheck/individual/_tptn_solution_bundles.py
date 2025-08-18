"""
This module implements some bundles of tp, tn solutions for scores
"""

from ._tptn_solutions import (
    acc_tn,
    acc_tp,
    bacc_tn,
    bacc_tp,
    bm_tn,
    bm_tp,
    dor_tn,
    dor_tp,
    f1n_tn,
    f1n_tp,
    f1p_tn,
    f1p_tp,
    fbn_tn,
    fbn_tp,
    fbp_tn,
    fbp_tp,
    fm_tn,
    fm_tp,
    gm_tn,
    gm_tp,
    ji_tn,
    ji_tp,
    kappa_tn,
    kappa_tp,
    lrn_tn,
    lrn_tp,
    lrp_tn,
    lrp_tp,
    mcc_tn,
    mcc_tp,
    mk_tn,
    mk_tp,
    npv_tn,
    npv_tp,
    ppv_tn,
    ppv_tp,
    pt_tn,
    pt_tp,
    sens_tp,
    spec_tn,
    upm_tn,
    upm_tp,
)

__all__ = ["tp_solutions", "tn_solutions", "tptn_solutions", "is_applicable_tptn"]


def is_applicable_tptn(score: str, value: float, to_compute: str) -> bool:
    """
    Checks if the tp-tn solution is computable

    Args:
        score (str): the score to check
        value (float): the value of the score
        to_compute (str): the figure to compute ('tp'/'tn')

    Returns:
        bool: True if the setup can be solved, False otherwise
    """
    if score == "pt" and value == 0.0 and to_compute == "tp":
        return False
    return not (score == "pt" and value == 1.0 and to_compute == "tn")


tp_solutions = {
    "mcc": mcc_tp,
    "acc": acc_tp,
    "sens": sens_tp,
    "ppv": ppv_tp,
    "npv": npv_tp,
    "fbp": fbp_tp,
    "f1p": f1p_tp,
    "fbn": fbn_tp,
    "f1n": f1n_tp,
    "upm": upm_tp,
    "gm": gm_tp,
    "fm": fm_tp,
    "mk": mk_tp,
    "lrp": lrp_tp,
    "lrn": lrn_tp,
    "bm": bm_tp,
    "pt": pt_tp,
    "dor": dor_tp,
    "ji": ji_tp,
    "bacc": bacc_tp,
    "kappa": kappa_tp,
}

tn_solutions = {
    "mcc": mcc_tn,
    "acc": acc_tn,
    "spec": spec_tn,
    "ppv": ppv_tn,
    "npv": npv_tn,
    "fbp": fbp_tn,
    "f1p": f1p_tn,
    "fbn": fbn_tn,
    "f1n": f1n_tn,
    "upm": upm_tn,
    "gm": gm_tn,
    "fm": fm_tn,
    "mk": mk_tn,
    "lrp": lrp_tn,
    "lrn": lrn_tn,
    "bm": bm_tn,
    "pt": pt_tn,
    "dor": dor_tn,
    "ji": ji_tn,
    "bacc": bacc_tn,
    "kappa": kappa_tn,
}

tptn_solutions = {
    key: {"tp": tp_solutions.get(key), "tn": tn_solutions.get(key)}
    for key in [
        "mcc",
        "acc",
        "spec",
        "sens",
        "ppv",
        "npv",
        "fbp",
        "f1p",
        "fbn",
        "f1n",
        "upm",
        "gm",
        "fm",
        "mk",
        "lrp",
        "lrn",
        "bm",
        "pt",
        "dor",
        "ji",
        "bacc",
        "kappa",
    ]
}
