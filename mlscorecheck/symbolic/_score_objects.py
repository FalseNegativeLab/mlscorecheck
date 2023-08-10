"""
This module implements the scores as objects representing some
additional algebraic properties of the scores beyond their formulation.
"""

import importlib

import numpy as np

from ..core import *
from ._algebra import *
from ..core import load_scores
from ..core import score_functions_standardized

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

scores = load_scores()
functions = score_functions_standardized()

class Score:
    """
    The Score base class
    """
    def __init__(self,
                    symbols,
                    descriptor,
                    *,
                    function,
                    symbol=None,
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
        self.descriptor = descriptor

        abbreviation = descriptor['abbreviation']
        name = descriptor['name']
        nans = descriptor.get('nans', None)
        synonyms = descriptor.get('synonyms', None)
        complement = descriptor.get('complement', None)
        args = descriptor.get('args_standardized')
        range_ = (descriptor.get('lower_bound', -np.inf), descriptor.get('upper_bound', -np.inf))

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
            'descriptor': self.descriptor,
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
                        scores['mcc'],
                        function=functions['mcc'])

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
                        scores['acc'],
                        function=functions['acc'])

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
                        scores['err'],
                        function=functions['err'])

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
                        scores['sens'],
                        function=functions['sens'])

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
                        scores['fnr'],
                        function=functions['fnr'])

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
                        scores['fpr'],
                        function=functions['fpr'])

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
                        scores['spec'],
                        function=functions['spec'])

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
                        scores['ppv'],
                        function=functions['ppv'])

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
                        scores['fdr'],
                        function=functions['fdr'])

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
                        scores['for_'],
                        function=functions['for_'])

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
                        scores['npv'],
                        function=functions['npv'])

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
                        scores['fbp'],
                        function=functions['fbp'])

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
                        scores['f1p'],
                        function=functions['f1p'])

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
                        scores['fbm'],
                        function=functions['fbm'])

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
                        scores['f1m'],
                        function=functions['f1m'])

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
                        scores['upm'],
                        function=functions['upm'])

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
                        scores['gm'],
                        function=functions['gm'])

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
                        scores['fm'],
                        function=functions['fm'])

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
                        scores['mk'],
                        function=functions['mk'])

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
                        scores['lrp'],
                        function=functions['lrp'])

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
                        scores['lrn'],
                        function=functions['lrn'])

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
                        scores['bm'],
                        function=functions['bm'])

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
                        scores['pt'],
                        function=functions['pt'])

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
                        scores['dor'],
                        function=functions['dor'])

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
                        scores['ji'],
                        function=functions['ji'])

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
                        scores['bacc'],
                        function=functions['bacc'])


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
                        scores['kappa'],
                        function=functions['kappa'])

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
                        scores['p4'],
                        function=functions['p4'])

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
