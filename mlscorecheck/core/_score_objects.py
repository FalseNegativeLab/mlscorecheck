"""
This module implements the scores as objects representing some
additional algebraic properties of the scores beyond their formulation.
"""

import importlib

import numpy as np

from ._scores import *
from ._algebra import *
from ._solutions import load_scores

__all__ = ['Score',
            'PositiveLikelihoodRatio',
            'MatthewsCorrelationCoefficient',
            'Accuracy',
            'ErrorRate',
            'Sensitivity',
            'FalseNegativeRate',
            'FalsePositiveRate',
            'Specificity',
            'PositivePredictiveValue',
            'NegativePredictiveValue',
            'FalseDiscoveryRate',
            'FalseOmissionRate',
            'FBetaPlus',
            'F1Plus',
            'FBetaMinus',
            'F1Minus',
            'UnifiedPerformanceMeasure',
            'GeometricMean',
            'FowlkesMallowsIndex',
            'Markedness',
            'PositiveLikelihoodRatio',
            'NegativeLikelihoodRatio',
            'Informedness',
            'PrevalenceThreshold',
            'DiagnosticOddsRatio',
            'JaccardIndex',
            'BalancedAccuracy',
            'CohensKappa',
            'P4',
            'get_base_objects']

class Score:
    """
    The Score base class
    """
    def __init__(self,
                    symbols,
                    *,
                    abbreviation,
                    name,
                    function,
                    nans=None,
                    range_=None,
                    symbol=None,
                    synonyms=None,
                    complement=None,
                    args=None,
                    expression=None,
                    equation=None,
                    equation_polynomial=None):
        """
        Constructor of the base class

        Args:
            symbols (Symbol): A Symbols object representing the base kit of symbols to use
            abbreviation (str): the abbreviation
            name (str): the name of the score
            function (callable): the functional form
            nans (list/None): the list of configurations when the score cannot be computed
            range_ (tuple/None): the lower and upper bound
            symbol (sympy/sage): the algebraic symbol
            synonyms (list/None): the synonyms if any
            complement (list/None): the complements
            args (list/None): the list of arguments
            expression (sympy/sage/None): the expression of the score
            equation (sympy/sage/None): the equation form
            equation_polynomial (sympy/sage/None): the equation in polynomial form
        """
        # setting the base kit of symbols
        self.symbols = symbols

        # setting the symbol
        if symbol is None:
            kwargs = {}
            if range_ is not None:
                kwargs['lower_bound'] = range_[0]
                kwargs['upper_bound'] = range_[1]
            self.symbol = self.symbols.algebra.create_symbol(abbreviation, real=True, **kwargs)
        else:
            self.symbol = symbol

        self.abbreviation = abbreviation
        self.name = name
        self.range = range_
        self.nans = nans

        # setting the score function
        if isinstance(function, str):
            module = importlib.import_module('mlscorecheck')
            self.function = getattr(module, function)
        else:
            self.function = function

        # generating the list of arguments
        if args is None:
            self.args = list(function.__code__.co_varnames[:function.__code__.co_kwonlyargcount])
        else:
            self.args = args

        # handling the case of sqrt expressions
        self.sqrt = 'sqrt' in self.args
        self.args = [arg for arg in self.args if arg != 'sqrt']

        self.synonyms = synonyms
        self.complement = complement

        # generating the list of argument symbols
        arg_symbols = {arg: getattr(symbols, arg) for arg in self.args}

        # setting the expression
        if expression is not None:
            self.expression = eval(expression, arg_symbols)
        else:
            self.expression = function(**arg_symbols)

        # setting the equation
        if equation is not None:
            self.equation = eval(equation, {**{self.abbreviation: self.symbol}, **arg_symbols})
        else:
            self.equation = self.symbol - self.expression

        # setting the polynomial equation
        if equation_polynomial is not None:
            self.equation_polynomial = eval(equation_polynomial, {**{self.abbreviation: self.symbol}, **arg_symbols})
        else:
            self.equation_polynomial = None

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
            'abbreviation': self.abbreviation,
            'name': self.name,
            'nans': self.nans,
            'range_': self.range,
            'synonyms': self.synonyms,
            'complement': self.complement,
            'args': self.args,
            'expression': str(self.expression),
            'equation': str(self.equation),
            'equation_polynomial': str(self.equation_polynomial),
            'function': self.function.__name__
        }

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
        Score.__init__(self,
                        symbols,
                        abbreviation='mcc',
                        name='matthews_correlation_coefficient',
                        range_=[-1, 1],
                        nans=[{'tp': 0, 'fp': 0},
                                {'tn': 0, 'fn': 0}],
                        function=matthews_correlation_coefficient_standardized)

        num, denom = symbols.algebra.num_denom(self.expression)
        self.equation_polynomial = symbols.algebra.simplify(self.symbol**2 * denom**2 - num**2)

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
        Score.__init__(self,
                        symbols,
                        abbreviation='acc',
                        name='accuracy',
                        range_=[0, 1],
                        nans=None,
                        function=accuracy_standardized,
                        complement='error_rate')

        _, denom = symbols.algebra.num_denom(self.expression)
        self.equation_polynomial = symbols.algebra.simplify(self.equation * denom)

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
        Score.__init__(self,
                        symbols,
                        abbreviation='err',
                        name='error_rate',
                        range_=[0, 1],
                        nans=None,
                        function=error_rate_standardized,
                        complement='accuracy')

        _, denom = symbols.algebra.num_denom(self.expression)
        self.equation_polynomial = symbols.algebra.simplify(self.equation * denom)

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
        Score.__init__(self,
                        symbols,
                        abbreviation='sens',
                        name='sensitivity',
                        range_=[0, 1],
                        nans=None,
                        function=sensitivity_standardized,
                        synonyms=['recall', 'true_positive_rate', 'Recall', 'TruePositiveRate'],
                        complement='false_negative_rate')

        _, denom = symbols.algebra.num_denom(self.expression)
        self.equation_polynomial = symbols.algebra.simplify(self.equation * denom)

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
        Score.__init__(self,
                        symbols,
                        abbreviation='fnr',
                        name='false_negative_rate',
                        range_=[0, 1],
                        nans=None,
                        function=false_negative_rate_standardized,
                        synonyms=['miss_rate', 'MissRate'],
                        complement='sensitivity')

        _, denom = symbols.algebra.num_denom(self.expression)
        self.equation_polynomial = symbols.algebra.simplify(self.equation * denom)

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
        Score.__init__(self,
                        symbols,
                        abbreviation='fpr',
                        name='false_positive_rate',
                        range_=[0, 1],
                        nans=None,
                        function=false_positive_rate_standardized,
                        synonyms=['false_alarm', 'fall_out', 'FalseAlarm', 'FallOut'],
                        complement='specificity')

        _, denom = symbols.algebra.num_denom(self.expression)
        self.equation_polynomial = symbols.algebra.simplify(self.equation * denom)

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
        Score.__init__(self,
                        symbols,
                        abbreviation='spec',
                        name='specificity',
                        range_=[0, 1],
                        nans=None,
                        function=specificity_standardized,
                        synonyms=['selectivity', 'Selectivity', 'true_negative_rate', 'TrueNegativeRate'],
                        complement='false_positive_rate')

        _, denom = symbols.algebra.num_denom(self.expression)
        self.equation_polynomial = symbols.algebra.simplify(self.equation * denom)

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
        Score.__init__(self,
                        symbols,
                        abbreviation='ppv',
                        name='positive_predictive_value',
                        range_=[0, 1],
                        nans=[{'tp': 0, 'fp': 0}],
                        function=positive_predictive_value_standardized,
                        synonyms=('precision', 'Precision'),
                        complement='false_discovery_rate')

        _, denom = symbols.algebra.num_denom(self.expression)
        self.equation_polynomial = symbols.algebra.simplify(self.equation * denom)

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
        Score.__init__(self,
                        symbols,
                        abbreviation='fdr',
                        name='false_discovery_rate',
                        range_=[0, 1],
                        nans=[{'tp': 0, 'fp': 0}],
                        function=false_discovery_rate_standardized,
                        complement='positive_predictive_value')

        _, denom = symbols.algebra.num_denom(self.expression)
        self.equation_polynomial = symbols.algebra.simplify(self.equation * denom)

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
        Score.__init__(self,
                        symbols,
                        abbreviation='for_',
                        name='false_omission_rate',
                        range_=[0, 1],
                        nans=[{'tn': 0, 'fn': 0}],
                        function=false_omission_rate_standardized,
                        complement='negative_predictive_value')

        _, denom = symbols.algebra.num_denom(self.expression)
        self.equation_polynomial = symbols.algebra.simplify(self.equation * denom)

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
        Score.__init__(self,
                        symbols,
                        abbreviation='npv',
                        name='negative_predictive_value',
                        range_=[0, 1],
                        nans=[{'tn': 0, 'fn': 0}],
                        function=negative_predictive_value_standardized,
                        complement='false_omission_rate')

        _, denom = symbols.algebra.num_denom(self.expression)
        self.equation_polynomial = symbols.algebra.simplify(self.equation * denom)

class FBetaPlus(Score):
    """
    The f-beta plus score
    """
    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self,
                        symbols,
                        abbreviation='fbp',
                        name='f_beta_plus',
                        range_=[0, np.inf],
                        nans=None,
                        function=f_beta_plus_standardized)

        _, denom = symbols.algebra.num_denom(self.expression)
        self.equation_polynomial = symbols.algebra.simplify(self.equation * denom)

class F1Plus(Score):
    """
    The f1-plus score
    """
    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self,
                        symbols,
                        abbreviation='f1p',
                        name='f1_plus',
                        range_=[0, 1],
                        function=f1_plus_standardized)

        _, denom = symbols.algebra.num_denom(self.expression)
        self.equation_polynomial = symbols.algebra.simplify(self.equation * denom)

class FBetaMinus(Score):
    """
    The f-beta minus score
    """
    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self,
                        symbols,
                        abbreviation='fbm',
                        name='f_beta_minus',
                        range_=[0, np.inf],
                        function=f_beta_minus_standardized)

        _, denom = symbols.algebra.num_denom(self.expression)
        self.equation_polynomial = symbols.algebra.simplify(self.equation * denom)

class F1Minus(Score):
    """
    The f1-minus score
    """
    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self,
                        symbols,
                        abbreviation='f1m',
                        name='f1_minus',
                        range_=[0, 1],
                        function=f1_minus_standardized)

        _, denom = symbols.algebra.num_denom(self.expression)
        self.equation_polynomial = symbols.algebra.simplify(self.equation * denom)

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
        Score.__init__(self,
                        symbols,
                        abbreviation='upm',
                        name='unified_performance_measure',
                        range_=[0, np.inf],
                        nans=[{'tp': 0, 'tn': 0}],
                        function=unified_performance_measure_standardized)

        _, denom = symbols.algebra.num_denom(self.expression)
        self.equation_polynomial = symbols.algebra.simplify(self.equation * denom)

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
        Score.__init__(self,
                        symbols,
                        abbreviation='gm',
                        name='geometric_mean',
                        range_=[0, 1],
                        function=geometric_mean_standardized)

        self.equation_polynomial = self.symbol**2 - symbols.tp**2*symbols.tn**2/(symbols.p**2*symbols.n**2)

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
        Score.__init__(self,
                        symbols,
                        abbreviation='fm',
                        name='fowlkes_mallows_index',
                        range_=[0, 1],
                        function=fowlkes_mallows_index_standardized)

        p = symbols.p
        n = symbols.n
        tp = symbols.tp
        tn = symbols.tn

        self.equation_polynomial = -(self.symbol**2*n*p - self.symbol**2*p*tn + self.symbol**2*p*tp - tp**2)

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
        Score.__init__(self,
                        symbols,
                        abbreviation='mk',
                        name='markedness',
                        range_=[0, 1],
                        function=markedness_standardized,
                        synonyms=['delta_p', 'DeltaP'])

        tp = symbols.tp
        tn = symbols.tn
        fp = symbols.n - tn
        fn = symbols.p - tp

        self.equation_polynomial = symbols.algebra.simplify(self.equation*(tp + fp)*(tn + fn))

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
        Score.__init__(self,
                        symbols,
                        abbreviation='lrp',
                        name='positive_likelihood_ratio',
                        range_=[0, np.inf],
                        nans=[{'fp': 0}],
                        function=positive_likelihood_ratio_standardized)

        _, denom = symbols.algebra.num_denom(self.expression)
        self.equation_polynomial = symbols.algebra.simplify(self.equation * denom)

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
        Score.__init__(self,
                        symbols,
                        abbreviation='lrn',
                        name='negative_likelihood_ratio',
                        range_=[0, np.inf],
                        nans=[{'fn': 0}],
                        function=negative_likelihood_ratio_standardized)

        _, denom = symbols.algebra.num_denom(self.expression)
        self.equation_polynomial = symbols.algebra.simplify(self.equation * denom)

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
        Score.__init__(self,
                        symbols,
                        abbreviation='bm',
                        name='informedness',
                        range_=[0, 1],
                        function=informedness_standardized,
                        synonyms=['bookmaker_informedness', 'BookmakerInformeness'])

        p = symbols.p
        n = symbols.n

        self.equation_polynomial = symbols.algebra.simplify(self.equation * p * n)

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
        Score.__init__(self,
                        symbols,
                        abbreviation='pt',
                        name='prevalence_threshold',
                        range_=[-np.inf, np.inf],
                        function=prevalence_threshold_standardized)

        tp = symbols.tp
        fp = symbols.n - symbols.tn
        p = symbols.p
        n = symbols.n

        self.equation_polynomial = (self.symbol * (tp/p - fp/n) + fp/n)**2 - fp*tp/(n*p)

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
        Score.__init__(self,
                        symbols,
                        abbreviation='dor',
                        name='diagnostic_odds_ratio',
                        range_=[0, np.inf],
                        nans=[{'tn': 'n'},
                                {'tp': 'p'}],
                        function=diagnostic_odds_ratio_standardized)

        tp = symbols.tp
        tn = symbols.tn
        p = symbols.p
        n = symbols.n

        self.equation_polynomial = symbols.algebra.simplify(self.symbol * (n - tn) * (p - tp) - tn*tp)

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
        Score.__init__(self,
                        symbols,
                        abbreviation='ji',
                        name='jaccard_index',
                        range_=[0, np.inf],
                        function=jaccard_index_standardized,
                        synonyms=['threat_score', 'ThreadScore', 'critical_success_index', 'CriticalSuccessIndex'])

        _, denom = symbols.algebra.num_denom(self.expression)
        self.equation_polynomial = symbols.algebra.simplify(self.equation * denom)

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
        Score.__init__(self,
                        symbols,
                        abbreviation='ba',
                        name='balanced_accuracy',
                        range_=[0, 1],
                        function=balanced_accuracy_standardized)


        tp = symbols.tp
        tn = symbols.tn
        p = symbols.p
        n = symbols.n

        self.equation_polynomial = symbols.algebra.simplify(self.symbol * 2 * n * p - (n*tp + p*tn))

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
        Score.__init__(self,
                        symbols,
                        abbreviation='kappa',
                        name='cohens_kappa',
                        range_=[-np.inf, np.inf],
                        nans=[{'tn': 0, 'tp': 0}],
                        function=cohens_kappa_standardized)

        tp = symbols.tp
        tn = symbols.tn
        p = symbols.p
        n = symbols.n

        self.equation_polynomial = symbols.algebra.simplify(self.symbol * (n**2 + n*tn - n*tp + p**2 - p*tn + p*tp) + 2 * (n*p - n*tn - p*tp))

class P4(Score):
    """
    The P4 score
    """
    def __init__(self, symbols):
        """
        The constructor of the score

        Args:
            symbols (Symbols): the algebraic symbols to be used
        """
        Score.__init__(self,
                        symbols,
                        abbreviation='p4',
                        name='p4',
                        range_=[0, 1],
                        nans=[{'tn': 0, 'tp': 0}],
                        function=p4_standardized)

        tp = symbols.tp
        tn = symbols.tn
        fp = symbols.n - tn
        fn = symbols.p - tp

        self.equation_polynomial = symbols.algebra.simplify(self.symbol * (4*tp*tn + (tp + tn)*(fp + fn)) - 4*tp*tn)

def get_base_objects(algebraic_system='sympy'):
    """
    Returns the dict of basic score objects

    Returns:
        dict: the dictionary of basic score objects
    """
    symbols = Symbols(algebraic_system=algebraic_system)
    score_objects = [cls(symbols=symbols) for cls in [Accuracy,
                                                        Sensitivity,
                                                        Specificity,
                                                        PositivePredictiveValue,
                                                        NegativePredictiveValue,
                                                        BalancedAccuracy,
                                                        F1Plus]]
    score_objects = {obj.abbreviation: obj for obj in score_objects}

    return score_objects
