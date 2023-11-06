"""
This module implements some bundles of tp, tn solutions for scores
"""

from ._tptn_solutions import (
    mcc_tp,
    mcc_tn,
    acc_tp,
    acc_tn,
    sens_tp,
    spec_tn,
    ppv_tp,
    ppv_tn,
    npv_tp,
    npv_tn,
    fbp_tp,
    fbp_tn,
    f1p_tp,
    f1p_tn,
    fbn_tp,
    fbn_tn,
    f1n_tp,
    f1n_tn,
    upm_tp,
    upm_tn,
    gm_tp,
    gm_tn,
    fm_tp,
    fm_tn,
    mk_tp,
    mk_tn,
    lrp_tp,
    lrp_tn,
    lrn_tp,
    lrn_tn,
    bm_tp,
    bm_tn,
    pt_tp,
    pt_tn,
    dor_tp,
    dor_tn,
    ji_tp,
    ji_tn,
    bacc_tp,
    bacc_tn,
    kappa_tp,
    kappa_tn,
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
    if score == "pt" and value == 1.0 and to_compute == "tn":
        return False

    return True


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
