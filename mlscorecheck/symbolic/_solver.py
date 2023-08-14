"""
This module implements the problem solver
"""

from ..core import logger, Solutions

__all__ = ['ProblemSolver',
            'collect_denominators_and_bases',
            '_collect_denominators_and_bases']

def _collect_denominators_and_bases(expression, denoms, bases, algebra):
    """
    Recursive core of collecting all denominators and bases

    Args:
        expression (sympy/sage): an expression
        denoms (list): the list of already collected denominators
        bases (list): the list of already collected bases
        algebra (Algebra): the algebra to be used
    """
    print('processing', expression)
    #simplified = algebra.simplify(expression)

    if algebra.is_division(expression):
        num, denom = algebra.num_denom(expression)
        num = algebra.simplify(num)
        denom = algebra.simplify(denom)
        print('nd', num, denom)

        if not algebra.is_trivial(denom):
            denoms.append(denom)
            _collect_denominators_and_bases(denom, denoms, bases, algebra)
        _collect_denominators_and_bases(num, denoms, bases, algebra)
    elif algebra.is_root(expression):
        # fractional exponents are already checked here
        base, _ = algebra.operands(expression)
        bases.append(base)
        print('base', base)

        for operand in algebra.operands(base):
            _collect_denominators_and_bases(operand, denoms, bases, algebra)
    else:
        for operand in algebra.operands(expression):
            if not algebra.is_trivial(operand):
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
        self.real_solutions = []

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

                sol = {var0: {'expression': algebra.simplify(v0_sol),
                                'symbols': algebra.free_symbols(v0_sol)},
                        var1: {'expression': algebra.simplify(v1[sym1]),
                                'symbols': algebra.free_symbols(v1[sym1])}}

                flag = True
                if str(sol[var0]['expression']) in [0, 'n', 'p']:
                    flag = False
                if str(sol[var1]['expression']) in [0, 'n', 'p']:
                    flag = False

                if flag:
                    self.solutions.append(sol)

        return self

    def edge_cases(self):
        """
        Collects the edge cases and populates the fields 'denoms' and 'bases'
        """
        self.denoms = []
        self.bases = []
        self.str_solutions = []

        for solution in self.solutions:
            denoms_sol = set()
            bases_sol = set()

            for item, sol in solution.items():
                print('aaa', sol['expression'])
                print('bbb', self.score0.symbols.algebra.simplify(sol['expression']))
                denoms, bases = collect_denominators_and_bases(self.score0.symbols.algebra.simplify(sol['expression']), self.score0.symbols.algebra)
                denoms = list(denoms)
                bases = list(bases)
                print('ccc', denoms)
                print('ddd', bases)
                denoms_sol = denoms_sol.union(set(denoms))
                bases_sol = bases_sol.union(set(bases))

            denoms_sol = [{'expression': str(denom), 'symbols': self.score0.symbols.algebra.free_symbols(denom)} for denom in denoms_sol]
            bases_sol = [{'expression': str(base), 'symbols': self.score0.symbols.algebra.free_symbols(base)} for base in bases_sol]

            print('EEE', denoms_sol)
            print('FFF', bases_sol)

            self.denoms.append(denoms_sol)
            self.bases.append(bases_sol)
            self.str_solutions.append({str(key): {key2: str(value2) if key2 == 'expression' else value2 for key2, value2 in value.items()} for key, value in solution.items()})

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
