"""
This module implements the scores as objects representing some
additional algebraic properties of the scores beyond their formulation.
"""

import importlib

import numpy as np

from ..core import safe_eval
from ..scores import score_specifications
from ..scores import score_functions_standardized_with_complements
from ..scores import score_functions_without_complements

from ._symbols import Symbols

__all__ = [
    "Score",
    "PositiveLikelihoodRatio",
    "MatthewsCorrelationCoefficient",
    "Accuracy",
    "ErrorRate",
    "Sensitivity",
    "FalseNegativeRate",
    "FalsePositiveRate",
    "Specificity",
    "PositivePredictiveValue",
    "NegativePredictiveValue",
    "FalseDiscoveryRate",
    "FalseOmissionRate",
    "FBetaPositive",
    "F1Positive",
    "FBetaNegative",
    "F1Negative",
    "UnifiedPerformanceMeasure",
    "GeometricMean",
    "FowlkesMallowsIndex",
    "Markedness",
    "PositiveLikelihoodRatio",
    "NegativeLikelihoodRatio",
    "Informedness",
    "PrevalenceThreshold",
    "DiagnosticOddsRatio",
    "JaccardIndex",
    "BalancedAccuracy",
    "CohensKappa",
    "get_base_objects",
    "get_all_objects",
    "get_objects_without_complements",
]

scores = score_specifications
functions = score_functions_standardized_with_complements


class Score:  # pylint: disable=too-many-instance-attributes
    """
    The Score base class
    """

    def __init__(
        self,
        symbols: Symbols,
        descriptor: dict,
        *,
        function,
        expression: str = None,
        equation: str = None
    ):
        """
        Constructor of the base class

        Args:
            symbols (Symbols): A Symbols object representing the base kit of symbols to use
            descriptor (dict): a dictionary descriptor of the score
            function (callable): the functional form
            expression (sympy_obj|sage_obj/None): the expression of the score
            equation (sympy_obj|sage_obj/None): the equation form
            equation_polynomial (sympy_obj|sage_obj/None): the equation in polynomial form
        """
        self.descriptor = descriptor

        self.abbreviation = descriptor["abbreviation"]
        self.name = descriptor["name"]
        self.nans = descriptor.get("nans")
        self.synonyms = descriptor.get("synonyms")
        self.complement = descriptor.get("complement")
        self.args = descriptor.get("args_standardized")
        self.range = (
            descriptor.get("lower_bound", -np.inf),
            descriptor.get("upper_bound", np.inf),
        )
        self.sqrt = descriptor.get("sqrt", False)

        # setting the base kit of symbols

        self.symbols = symbols

        # setting the symbol
        kwargs = {}
        if self.range[0] > -np.inf:
            kwargs["lower_bound"] = self.range[0]
        if self.range[1] < np.inf:
            kwargs["upper_bound"] = self.range[1]
        self.symbol = self.symbols.algebra.create_symbol(
            self.abbreviation, real=True, **kwargs
        )

        # setting the score function
        if isinstance(function, str):
            module = importlib.import_module("mlscorecheck.scores")
            self.function = getattr(module, function)
        else:
            self.function = function

        # generating the list of argument symbols
        arg_symbols = {arg: getattr(symbols, arg) for arg in self.args}

        if self.sqrt:
            arg_symbols["sqrt"] = symbols.sqrt

        # setting the expression
        if expression is not None:
            self.expression = safe_eval(expression, arg_symbols)
        else:
            self.expression = function(**arg_symbols)

        # setting the equation
        if equation is not None:
            subs = {**{self.abbreviation: self.symbol}, **arg_symbols}
            self.equation = safe_eval(equation, subs)
        else:
            self.equation = self.symbol - self.expression

        # setting the polynomial equation
        subs = {**{self.abbreviation: self.symbol}, **arg_symbols}
        self.equation_polynomial = safe_eval(descriptor["polynomial_equation"], subs)

    def get_algebra(self):
        """
        Return the algebra behind the symbols
        """

        return self.symbols.algebra

    def to_dict(self):
        """
        Returns a dictionary representation

        Returns:
            dict: the dictionary representation
        """
        return {
            "descriptor": self.descriptor,
            "expression": str(self.expression),
            "equation": str(self.equation),
            "function": self.function.__name__,
        }


class Accuracy(Score):
    """
    The accuracy score
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["acc"], function=functions["acc"])


class ErrorRate(Score):
    """
    The error rate score
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["err"], function=functions["err"])


class Sensitivity(Score):
    """
    The sensitivity score
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["sens"], function=functions["sens"])


class FalseNegativeRate(Score):
    """
    The false negative rate
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["fnr"], function=functions["fnr"])


class FalsePositiveRate(Score):
    """
    The false positive rate
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["fpr"], function=functions["fpr"])


class Specificity(Score):
    """
    The specificity score
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["spec"], function=functions["spec"])


class PositivePredictiveValue(Score):
    """
    The positive predictive value
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["ppv"], function=functions["ppv"])


class FalseDiscoveryRate(Score):
    """
    The false discovery rate
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["fdr"], function=functions["fdr"])


class FalseOmissionRate(Score):
    """
    The false omission rate
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["for_"], function=functions["for_"])


class NegativePredictiveValue(Score):
    """
    The negative predictive value
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["npv"], function=functions["npv"])


class FBetaPositive(Score):
    """
    The f-beta plus score
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["fbp"], function=functions["fbp"])


class F1Positive(Score):
    """
    The f1-plus score
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["f1p"], function=functions["f1p"])


class FBetaNegative(Score):
    """
    The f-beta minus score
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["fbn"], function=functions["fbn"])


class F1Negative(Score):
    """
    The f1-minus score
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["f1n"], function=functions["f1n"])


class GeometricMean(Score):
    """
    The geometric mean score
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["gm"], function=functions["gm"])


class FowlkesMallowsIndex(Score):
    """
    The Fowlkes-Mallows-index
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["fm"], function=functions["fm"])


class UnifiedPerformanceMeasure(Score):
    """
    The unified performance measure score
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["upm"], function=functions["upm"])


class Markedness(Score):
    """
    The markedness score
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["mk"], function=functions["mk"])


class PositiveLikelihoodRatio(Score):
    """
    The positive likelihood ratio score
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["lrp"], function=functions["lrp"])


class NegativeLikelihoodRatio(Score):
    """
    The negative likelihood ratio score
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["lrn"], function=functions["lrn"])


class Informedness(Score):
    """
    The informedness score
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["bm"], function=functions["bm"])


class PrevalenceThreshold(Score):
    """
    The prevalence threshold
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["pt"], function=functions["pt"])


class DiagnosticOddsRatio(Score):
    """
    The diagnostic odds ratio
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["dor"], function=functions["dor"])


class JaccardIndex(Score):
    """
    The Jaccard index
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["ji"], function=functions["ji"])


class BalancedAccuracy(Score):
    """
    The balanced accuracy score
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["bacc"], function=functions["bacc"])


class CohensKappa(Score):
    """
    Cohen's kappa
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["kappa"], function=functions["kappa"])


class MatthewsCorrelationCoefficient(Score):
    """
    The MatthewsCorrelationCoefficient score
    """

    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self, symbols, scores["mcc"], function=functions["mcc"])


def get_base_objects(algebraic_system: str = "sympy") -> dict:
    """
    Returns the dict of basic score objects

    Args:
        algebraic_system (str): 'sympy'/'sage' - the algebraic system to use

    Returns:
        dict: the dictionary of basic score objects
    """
    symbols = Symbols(algebraic_system=algebraic_system)
    score_objects = [
        cls(symbols=symbols)
        for cls in [
            Accuracy,
            Sensitivity,
            Specificity,
            PositivePredictiveValue,
            NegativePredictiveValue,
            BalancedAccuracy,
            F1Positive,
            FowlkesMallowsIndex,
        ]
    ]
    score_objects = {obj.abbreviation: obj for obj in score_objects}

    return score_objects


def get_all_objects(algebraic_system: str = "sympy") -> dict:
    """
    Returns the dict of all score objects

    Args:
        algebraic_system (str): 'sympy'/'sage' - the algebraic system to use

    Returns:
        dict: the dictionary of all score objects
    """
    symbols = Symbols(algebraic_system=algebraic_system)
    score_objects = [cls(symbols=symbols) for cls in Score.__subclasses__()]
    score_objects = {obj.abbreviation: obj for obj in score_objects}

    return score_objects


def get_objects_without_complements(algebraic_system: str = "sympy") -> dict:
    """
    Returns the dict of basic score objects without complements

    Args:
        algebraic_system (str): 'sympy'/'sage' - the algebraic system to use

    Returns:
        dict: the dictionary of score objects
    """
    symbols = Symbols(algebraic_system=algebraic_system)
    score_objects = [cls(symbols=symbols) for cls in Score.__subclasses__()]
    score_objects = {
        obj.abbreviation: obj
        for obj in score_objects
        if obj.abbreviation in score_functions_without_complements
    }

    return score_objects
