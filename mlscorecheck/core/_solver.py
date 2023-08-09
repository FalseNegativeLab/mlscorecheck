"""
This module implements the problem solver
"""

from ._logger import logger
from ._interval import Interval, IntervalUnion

__all__ = ['ProblemSolver', 'Solution', 'Solutions']

class ZeroDivision(Exception):
    """
    An exception indicating zero division
    """
    def __init__(self, expression):
        """
        The constructor of the exception

        Args:
            expression (dict): the expression and its value
        """
        self.expression = expression

class NegativeBase(BaseException):
    """
    An exception indicating the root of a negative number
    """
    def __init__(self, expression):
        """
        The constructor of the exception

        Args:
            expression (dict): the expression and its value
        """
        self.expression = expression

def _collect_denominators_and_bases(expression, denoms, bases, algebra):
    """
    Recursive core of collecting all denominators and bases

    Args:
        expression (sympy/sage): an expression
        denoms (list): the list of already collected denominators
        bases (list): the list of already collected bases
        algebra (Algebra): the algebra to be used
    """
    num, denom = algebra.num_denom(algebra.simplify(expression))

    if not algebra.is_trivial(denom):
        denoms.append(denom)
        _collect_denominators_and_bases(denom, denoms, bases, algebra)
    else:
        pass
    if algebra.is_root(num):
        # fractional exponents are already checked here
        base, _ = algebra.operands()
        bases.append(base)
        _collect_denominators_and_bases(base, denoms, bases, algebra)
    else:
        for operand in algebra.operands(num):
            _collect_denominators_and_bases(operand, denoms, bases, algebra)
    return

def collect_denominators_and_bases(expression, algebra):
    """
    Top level function for recursively collecting all denominators and bases

    Args:
        expression (sympy/sage): the expression to process
        algebra (Algebra): the algebra to be used

    Returns:
        list, list: the collected denominators and bases
    """
    denoms = []
    bases = []
    _collect_denominators_and_bases(expression, denoms, bases, algebra)
    return denoms, bases

class Solution:
    """
    Represents one single solution (expressions for tp and tn) and corresponding
    non-zero and non-negative conditions as expressions
    """
    def __init__(self,
                    solution,
                    non_zero,
                    non_negative):
        """
        Constructor of the solution

        Args:
            solution (dict(dict)): the solutions ({'expressions': , 'symbols': })
            non_zero (list(dict)): the non-zero conditions ([{'expression': , 'symbols':}])
            non_negative (list(dict)): the non-negative conditions ([{'expression': , 'symbols':}])
        """
        self.solution = solution
        self.non_zero = non_zero
        self.non_negative = non_negative

        #print(self.solution)
        #print(self.non_zero)
        #print(self.non_negative)

        # extracting all symbols
        self.all_symbols = set()

        for _, item in self.solution['symbols'].items():
            self.all_symbols = self.all_symbols.union(set(item))

        for non_zero in self.non_zero:
            self.all_symbols = self.all_symbols.union(non_zero['symbols'])

        for non_negative in self.non_negative:
            self.all_symbols = self.all_symbols.union(non_negative['symbols'])

        #print(self.all_symbols)

    def to_dict(self):
        """
        Returning a dictionary representation

        Returns:
            dict: the dictionary representation
        """
        return {'solution': self.solution,
                'non_zero': self.non_zero,
                'non_negative': self.non_negative}

    def check_non_zeros(self, evals):
        for key, value in evals.items():
            if isinstance(value, (Interval, IntervalUnion)):
                if value.contains(0):
                    return {key: value}
            else:
                if value == 0:
                    return {key: value}
        return None

    def non_zero_conditions(self, subs):
        """
        Checking the non-zero condition with a substitution

        Args:
            subs (dict): a substitution

        Returns:
            bool: the result of the check
        """
        non_zeros = {non_zero['expression']: eval(non_zero['expression'], subs)
                                for non_zero in self.non_zero}

        #print(non_zeros)

        return self.check_non_zeros(non_zeros)

    def check_non_negatives(self, evals):
        for key, value in evals.items():
            if isinstance(value, (Interval, IntervalUnion)):
                if isinstance(value, Interval):
                    if value.lower_bound < 0:
                        return {key: value}
                else:
                    if any(interval.lower_bound < 0 for interval in value.intervals):
                        return {key: value}
            else:
                if value < 0:
                    return {key: value}
        return None

    def non_negative_conditions(self, subs):
        """
        Checking the non-negativity condition with a substitution

        Args:
            subs (dict): a substitution

        Returns:
            bool: the result of the check
        """
        non_negatives = {non_negative['expression']: eval(non_negative['expression'], subs)
                                for non_negative in self.non_negative}

        return self.check_non_negatives(non_negatives)

    def evaluate(self, subs):
        """
        Evaluate the solution with a substitution

        Args:
            subs (dict): a substitution

        Returns:
            dict: the results

        Throws:
            ZeroDivision: when zero division occurs
            NegativeBase: when the root of a negative number is taken
        """
        subs = {key: subs[key] for key in self.all_symbols}

        non_zero = self.non_zero_conditions(subs)

        if non_zero:
            raise ZeroDivision(expression=non_zero)

        non_negative = self.non_negative_conditions(subs)

        if non_negative:
            raise NegativeBase(expression=non_negative)

        return {key: eval(value, subs) for key, value in self.solution['expressions'].items()}

class Solutions:
    """
    Represents all solutions to a particular problem
    """
    def __init__(self,
                    scores,
                    solutions):
        """
        The constructor of the object

        Args:
            scores (list): the list of the score descriptors
            solutions (list): the list of the individual solutions
        """
        self.scores = scores
        self.solutions = [Solution(**sol) for sol in solutions]

    def to_dict(self):
        """
        Returns a dictionary representation

        Returns:
            dict: the dictionary representation
        """

        return {'scores': self.scores,
                'solutions': [sol.to_dict() for sol in self.solutions]}

    def evaluate(self, subs):
        """
        Evaluate the solutions with a substitution

        Args:
            subs (dict): a substitution

        Returns:
            dict: the results
        """
        results = []

        for sol in self.solutions:
            try:
                res = sol.evaluate(subs)
                res['message'] = None
            except ZeroDivision as exc:
                res = {'tp': None,
                        'tn': None,
                        'message': 'zero division error',
                        'denominator': exc.expression}
            except NegativeBase as exc:
                res = {'tp': None,
                        'tn': None,
                        'message': 'negative base',
                        'base': exc.expression}
            tmp = {'formula': sol.solution['expressions'],
                    'results': res}
            results.append(tmp)

        return results

class ProblemSolver:
    """
    The problem solver object, used to solve the individual problems with
    the aid of computer algebra systems
    """
    def __init__(self, score0, score1):
        """
        The constructor of the object

        Args:
            score0 (ScoreObject): the first score object
            score1 (ScoreObject): the second score object
        """
        self.score0 = score0
        self.score1 = score1

    def solve(self):
        """
        Solves the problem

        Returns:
            self: the solver object
        """
        self.solutions = []

        equation0 = self.score0.equation_polynomial
        equation1 = self.score1.equation_polynomial

        var0 = 'tp'
        var1 = 'tn'

        args0 = self.score0.args
        args1 = self.score1.args

        n_vars0 = int('tp' in args0) + int('tn' in args0)
        n_vars1 = int('tp' in args1) + int('tn' in args1)

        if n_vars0 != 1 and n_vars1 == 1:
            equation0, equation1 = self.score1.equation_polynomial, self.score0.equation_polynomial
            args0, args1 = args1, args0

        if 'tp' not in args1:
            var0, var1 = 'tn', 'tp'

        if var0 not in args0:
            var0, var1 = var1, var0

        sym0 = getattr(self.score0.symbols, var0)
        sym1 = getattr(self.score0.symbols, var1)

        algebra = self.score0.symbols.algebra

        logger.info(f'eq0 {equation0}')
        logger.info(f'sym0 {sym0}')

        v0s = algebra.solve(equation0, sym0)

        logger.info(f'v0s {v0s}')

        for v0 in v0s:
            logger.info('v0 {v0}')
            equation1_tmp = equation1

            logger.info(f'{algebra.args(equation1)}')
            if sym0 in algebra.args(equation1):
                logger.info(f'substitution {equation1} {v0}')
                equation1_tmp = algebra.subs(equation1, v0)

            logger.info(f'eq1tmp {equation1_tmp}')

            v1s = algebra.solve(equation1_tmp, sym1)

            logger.info(f'v1s {v1s}')

            for v1 in v1s:
                v0_sol = v0[sym0]

                if sym1 in algebra.args(v0_sol):
                    v0_sol = algebra.subs(v0_sol, v1)

                sol = {}

                sol[var0] = algebra.simplify(v0_sol)
                sol[var1] = algebra.simplify(v1[sym1])

                sol_symbols = {}
                sol_symbols[var0] = algebra.free_symbols(sol[var0])
                sol_symbols[var1] = algebra.free_symbols(sol[var1])

                self.solutions.append({'expressions': {key: str(item) for key, item in sol.items()}, 'symbols': sol_symbols})

        return self

    def edge_cases(self):
        """
        Collects the edge cases and populates the fields 'denoms' and 'bases'
        """
        self.denoms = []
        self.bases = []

        for solution in self.solutions:
            denoms_sol = set()
            bases_sol = set()

            for _, item in solution['expressions'].items():
                denoms, bases = collect_denominators_and_bases(item, self.score0.symbols.algebra)
                denoms = [denom for denom in denoms]
                bases = [base for base in bases]
                denoms_sol = denoms_sol.union(set(denoms))
                bases_sol = bases_sol.union(set(bases))

            denoms_sol = [{'expression': str(denom), 'symbols': self.score0.symbols.algebra.free_symbols(denom)} for denom in denoms]
            bases_sol = [{'expression': str(base), 'symbols': self.score0.symbols.algebra.free_symbols(base)} for base in bases]

            self.denoms.append(denoms_sol)
            self.bases.append(bases_sol)

    def get_solution(self):
        """
        Transforms the solution into a solution object

        Returns:
            Solution: the solution object
        """
        results = []
        for solution, denoms, bases in zip(self.solutions, self.denoms, self.bases):
            sol_str = {str(key): item for key, item in solution.items()}
            denom_str = [(item) for item in denoms]
            base_str = [(item) for item in bases]

            results.append({'solution': sol_str,
                            'non_zero': denom_str,
                            'non_negative': base_str})

        solution = Solutions(scores=[self.score0.to_dict(), self.score1.to_dict()],
                            solutions=results)

        return solution
