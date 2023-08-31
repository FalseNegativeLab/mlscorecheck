"""
This module implements the problem solver
"""

from ..core import logger
from ..individual import Solutions

__all__ = ['ProblemSolver',
            'collect_denominators_and_bases',
            '_collect_denominators_and_bases']

def _collect_denominators_and_bases(expression, denoms, bases, algebra):
    """
    Recursive core of collecting all denominators and bases

    Args:
        expression (sympy_obj|sage_obj): an expressio
        denoms (list): the list of already collected denominators
        bases (list): the list of already collected bases
        algebra (Algebra): the algebra to be used
    """

    if algebra.is_division(expression):
        num, denom = algebra.num_denom(expression)
        num = algebra.simplify(num)
        denom = algebra.simplify(denom)

        if not algebra.is_trivial(denom):
            denoms.append(denom)
            _collect_denominators_and_bases(denom, denoms, bases, algebra)
        _collect_denominators_and_bases(num, denoms, bases, algebra)
    elif algebra.is_root(expression):
        # fractional exponents are already checked here
        base, _ = algebra.operands(expression)
        bases.append(base)

        for operand in algebra.operands(base):
            _collect_denominators_and_bases(operand, denoms, bases, algebra)
    else:
        for operand in algebra.operands(expression):
            if not algebra.is_trivial(operand):
                _collect_denominators_and_bases(operand, denoms, bases, algebra)

def collect_denominators_and_bases(expression, algebra):
    """
    Top level function for recursively collecting all denominators and bases

    Args:
        expression (sympy_obj|sage_obj): the expression to process
        algebra (Algebra): the algebra to be used

    Returns:
        list, list: the collected denominators and bases
    """
    denoms = []
    bases = []
    _collect_denominators_and_bases(expression, denoms, bases, algebra)
    return denoms, bases

class ProblemSolver:
    """
    The problem solver object, used to solve the individual problems with
    the aid of computer algebra systems
    """
    def __init__(self, score0, score1):
        """
        The constructor of the object

        Args:
            score0 (Score): the first score object
            score1 (Score): the second score object
        """
        self.score0 = score0
        self.score1 = score1
        self.solutions = None
        self.real_solutions = None
        self.denoms = None
        self.bases = None
        self.str_solutions = None

    def determine_variable_order(self):
        """
        Determine the variable order

        Returns:
            obj, obj, obj, obj: the first variable to solve; the second variable
                        to solve; the first equation to solve; the second equation to solve
        """
        equation0 = self.score0.equation_polynomial
        equation1 = self.score1.equation_polynomial

        # the initial choice of variable order to solve
        var0 = 'tp'
        var1 = 'tn'

        args0 = self.score0.args
        args1 = self.score1.args

        n_vars0 = int('tp' in args0) + int('tn' in args0)
        n_vars1 = int('tp' in args1) + int('tn' in args1)

        if n_vars0 != 1 and n_vars1 == 1:
            equation0, equation1 = (self.score1.equation_polynomial,
                                    self.score0.equation_polynomial)
            args0, args1 = args1, args0

        if 'tp' not in args1:
            var0, var1 = 'tn', 'tp'

        if var0 not in args0:
            var0, var1 = var1, var0

        return var0, var1, equation0, equation1

    def corner_case_solution(self, solution):
        """
        Checks if the solution is a corner case solution

        Args:
            solution (dict): a pair of solutions

        Returns:
            bool: a flag indicating if the solution is corner case
        """
        var0, var1 = list(solution.keys())
        flag = False
        if str(solution[var0]['expression']) in {'0', 'n', 'p'}:
            # triggered in the ppv-fm case
            flag = True
        if str(solution[var1]['expression']) in {'0', 'n', 'p'}:
            # triggered in the fm-ppv case
            flag = True

        return flag

    def solve(self):
        """
        Solves the problem

        Returns:
            self: the solver object
        """
        self.solutions = []
        self.real_solutions = []

        var0, var1, equation0, equation1 = self.determine_variable_order()

        # querying the symbols
        sym0 = getattr(self.score0.symbols, var0)
        sym1 = getattr(self.score0.symbols, var1)

        algebra = self.score0.symbols.algebra

        logger.info('eq0 %s', equation0)
        logger.info('sym0 %s', sym0)

        # solving the first equation for sym0
        v0_sols = algebra.solve(equation0, sym0)

        logger.info('v0s %s', v0_sols)

        for v0_sol in v0_sols:
            # iterating through all solutions
            logger.info('v0 %s', v0_sol)

            # substitution into the other equation if needed
            equation1_tmp = equation1

            logger.info('%s', algebra.args(equation1))
            if sym0 in algebra.args(equation1):
                logger.info('substitution %s %s', equation1, v0_sol)
                equation1_tmp = algebra.subs(equation1, v0_sol)

            logger.info('eq1tmp %s', equation1_tmp)

            # solving for sym1
            v1_sols = algebra.solve(equation1_tmp, sym1)

            logger.info('v1s %s', v1_sols)

            # assembling all solutions
            for v1_sol in v1_sols:
                v0_final = v0_sol[sym0]

                if sym1 in algebra.args(v0_final):
                    v0_final = algebra.subs(v0_final, v1_sol)

                sol = {var0: {'expression': algebra.simplify(v0_final),
                                'symbols': algebra.free_symbols(v0_final)},
                        var1: {'expression': algebra.simplify(v1_sol[sym1]),
                                'symbols': algebra.free_symbols(v1_sol[sym1])}}

                if not self.corner_case_solution(sol):
                    self.solutions.append(sol)

        return self

    def edge_cases(self):
        """
        Collects the edge cases and populates the fields 'denoms' and 'bases'
        """
        self.denoms = []
        self.bases = []
        self.str_solutions = []

        algebra = self.score0.symbols.algebra

        for solution in self.solutions:
            denoms_sol = set()
            bases_sol = set()

            for _, sol in solution.items():
                simplified = algebra.simplify(sol['expression'])
                denoms, bases = collect_denominators_and_bases(simplified, algebra)
                denoms = list(denoms)
                bases = list(bases)
                denoms_sol = denoms_sol.union(set(denoms))
                bases_sol = bases_sol.union(set(bases))

            denoms_sol = [{'expression': str(denom),
                            'symbols': algebra.free_symbols(denom)} for denom in denoms_sol]
            bases_sol = [{'expression': str(base),
                            'symbols': algebra.free_symbols(base)} for base in bases_sol]

            self.denoms.append(denoms_sol)
            self.bases.append(bases_sol)
            tmp = {str(key): {key2: str(value2)
                                if key2 == 'expression'
                                else value2 for key2, value2 in value.items()}
                    for key, value in solution.items()}
            self.str_solutions.append(tmp)

    def get_solution(self):
        """
        Transforms the solution into a solution object

        Returns:
            Solution: the solution object
        """
        results = []
        for solution, denoms, bases in zip(self.str_solutions, self.denoms, self.bases):
            results.append({'solution': solution,
                            'non_zero': denoms,
                            'non_negative': bases})

        solution = Solutions(scores=[self.score0.abbreviation, self.score1.abbreviation],
                            solutions=results)

        return solution
