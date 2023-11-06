"""
This module implements the problem solver
"""

from ..core import logger
from ..individual import Solutions
from ._algebra import Algebra
from ._score_objects import Score

__all__ = [
    "ProblemSolver",
    "collect_denominators_and_bases",
    "_collect_denominators_and_bases",
    "solve_order",
    "check_recurrent_solution",
]


def _collect_denominators_and_bases(
    expression, denoms: list, bases: list, algebra: Algebra, depth: int
):
    """
    Recursive core of collecting all denominators and bases

    Args:
        expression (sympy_obj|sage_obj): an expression
        denoms (list): the list of already collected denominators
        bases (list): the list of already collected bases
        algebra (Algebra): the algebra to be used
    """

    if algebra.is_division(expression):
        num, denom = algebra.num_denom(expression)
        num = algebra.simplify(num)
        denom = algebra.simplify(denom)

        if not algebra.is_trivial(denom):
            denoms.append((denom, depth))
            _collect_denominators_and_bases(denom, denoms, bases, algebra, depth + 1)
        _collect_denominators_and_bases(num, denoms, bases, algebra, depth + 1)
    elif algebra.is_root(expression):
        # fractional exponents are already checked here
        base, exponent = algebra.operands(expression)
        bases.append((base, float(exponent), depth))

        for operand in algebra.operands(base):
            _collect_denominators_and_bases(operand, denoms, bases, algebra, depth + 1)
    else:
        for operand in algebra.operands(expression):
            if not algebra.is_trivial(operand):
                _collect_denominators_and_bases(
                    operand, denoms, bases, algebra, depth + 1
                )


def collect_denominators_and_bases(expression, algebra: Algebra) -> (list, list):
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
    _collect_denominators_and_bases(expression, denoms, bases, algebra, depth=0)
    denom_dict = {}
    base_dict = {}

    for denom, depth in denoms:
        denom_str = str(denom)
        if denom_str in denom_dict:
            if denom_dict[denom_str][1] < depth:
                denom_dict[denom_str] = (denom, depth)
        else:
            denom_dict[denom_str] = (denom, depth)
    denoms = list(denom_dict.values())

    for base, exponent, depth in bases:
        base_str = str((base, exponent))
        if base_str in base_dict:
            if base_dict[base_str][2] < depth:
                base_dict[base_str] = (base, exponent, depth)
        else:
            base_dict[base_str] = (base, exponent, depth)
    bases = list(base_dict.values())

    return denoms, bases


def solve_order(score0: Score, score1: Score):
    """
    Determining the order the equations should be solved, based on using the shortest solution
    first

    Args:
        score0 (Score): the first score object
        score1 (Score): the second score object

    Returns:
        obj, obj, str, str: the equations to be solved and the order of
        the 'tp'/'tn' variables
    """
    symbols = score0.symbols
    algebra = score0.get_algebra()

    sols = {}
    lens = {}

    for figure in ["tp", "tn"]:
        for score_idx, score in enumerate([score0, score1]):
            key = (figure, score_idx)
            if figure in algebra.free_symbols(score.equation_polynomial):
                sols[key] = algebra.solve(
                    score.equation_polynomial, getattr(symbols, figure)
                )
                lens[key] = len(str(sols[key][0][getattr(symbols, figure)]))

    min_length = 1e10
    min_variable = None

    for key, value in lens.items():
        if value < min_length:
            min_length = value
            min_variable = key

    if ("tn", 0) not in sols or ("tp", 1) not in sols:
        return score0.equation_polynomial, score1.equation_polynomial, "tp", "tn"
    if ("tp", 0) not in sols or ("tn", 1) not in sols:
        return score0.equation_polynomial, score1.equation_polynomial, "tn", "tp"

    first_variable = None
    second_variable = None
    if min_variable[0] == "tp":
        first_variable = symbols.tp
        second_variable = symbols.tn
        if min_variable[1] == 1:
            score0, score1 = score1, score0
    elif min_variable[0] == "tn":
        first_variable = symbols.tn
        second_variable = symbols.tp
        if min_variable[1] == 1:
            score0, score1 = score1, score0

    return (
        score0.equation_polynomial,
        score1.equation_polynomial,
        str(first_variable),
        str(second_variable),
    )


def check_recurrent_solution(symbol: str, symbols: list) -> bool:
    """
    Checks and warns if the solution is recurrent in the variable

    Args:
        symbol (str): the variable to check
        symbols (list): the free symbols
    """
    if symbol in symbols:
        logger.warning("recurrent %s %s", symbol, str(symbols))

    return symbol in symbols


class ProblemSolver:
    """
    The problem solver object, used to solve the individual problems with
    the aid of computer algebra systems
    """

    def __init__(self, score0: Score, score1: Score):
        """
        The constructor of the object

        Args:
            score0 (Score): the first score object
            score1 (Score): the second score object
        """
        self.score0 = score0
        self.score1 = score1
        self.solutions = None
        self.denoms = None
        self.bases = None
        self.conditions = None
        self.str_solutions = None

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
        if str(solution[var0]["expression"]) in {"0", "n", "p"}:
            # triggered in the ppv-fm case
            flag = True
        if str(solution[var1]["expression"]) in {"0", "n", "p"}:
            # triggered in the fm-ppv case
            flag = True

        return flag

    def solve(self, **kwargs):
        """
        Solves the problem

        Args:
            kwargs (dict): additional parameters to the solver

        Returns:
            self: the solver object
        """
        logger.info("solving %s %s", self.score0.abbreviation, self.score1.abbreviation)

        self.solutions = []

        equation0, equation1, var0, var1 = solve_order(self.score0, self.score1)

        # querying the symbols
        sym0 = getattr(self.score0.symbols, var0)
        sym1 = getattr(self.score0.symbols, var1)

        algebra = self.score0.symbols.algebra

        for v0_sol in algebra.solve(equation0, sym0, **kwargs):
            # iterating through all solutions

            # substitution into the other equation if needed
            equation1_tmp = (
                algebra.subs(equation1, v0_sol)
                if sym0 in algebra.args(equation1)
                else equation1
            )

            # solving for sym1
            v1_sols = algebra.solve(equation1_tmp, sym1, **kwargs)

            logger.info("solved")

            # assembling all solutions
            for v1_sol in v1_sols:
                v0_final = (
                    algebra.subs(v0_sol[sym0], v1_sol)
                    if sym1 in algebra.args(v0_sol[sym0])
                    else v0_sol[sym0]
                )

                sol = {
                    var0: {
                        "expression": algebra.simplify(v0_final),
                        "symbols": algebra.free_symbols(v0_final),
                    },
                    var1: {
                        "expression": algebra.simplify(v1_sol[sym1]),
                        "symbols": algebra.free_symbols(v1_sol[sym1]),
                    },
                }

                check_recurrent_solution(str(var0), sol[var0]["symbols"])
                check_recurrent_solution(str(var1), sol[var1]["symbols"])

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
        self.conditions = []

        algebra = self.score0.symbols.algebra

        for solution in self.solutions:
            denoms_sol = {}
            bases_sol = {}

            for _, sol in solution.items():
                denoms, bases = collect_denominators_and_bases(
                    algebra.simplify(sol["expression"]), algebra
                )
                denoms = list(denoms)
                bases = list(bases)
                for denom, depth in denoms:
                    denom_str = str(denom)
                    if (denom_str not in denoms_sol) or (
                        denoms_sol[denom_str][1] < depth
                    ):
                        denoms_sol[denom_str] = (denom, depth)
                for base, exponent, depth in bases:
                    base_str = str((base, exponent))
                    if (base_str not in bases_sol) or (bases_sol[base_str][2] < depth):
                        bases_sol[base_str] = (base, exponent, depth)

            denoms_sol = [
                {
                    "expression": str(denom[0]),
                    "symbols": algebra.free_symbols(denom[0]),
                    "depth": denom[1],
                    "mode": "non-zero",
                }
                for denom in list(denoms_sol.values())
            ]
            bases_sol = [
                {
                    "expression": str(base[0]),
                    "exponent": base[1],
                    "symbols": algebra.free_symbols(base[0]),
                    "depth": base[2],
                    "mode": "non-negative",
                }
                for base in list(bases_sol.values())
            ]

            self.denoms.append(denoms_sol)
            self.bases.append(bases_sol)
            self.conditions.append(denoms_sol + bases_sol)
            tmp = {
                str(key): {
                    key2: str(value2) if key2 == "expression" else value2
                    for key2, value2 in value.items()
                }
                for key, value in solution.items()
            }
            self.str_solutions.append(tmp)

    def get_solution(self) -> Solutions:
        """
        Transforms the solution into a solution object

        Returns:
            Solution: the solution object
        """
        results = []

        for solution, conditions in zip(self.str_solutions, self.conditions):
            results.append({"solution": solution, "conditions": conditions})

        solution = Solutions(
            scores=[self.score0.abbreviation, self.score1.abbreviation],
            solutions=results,
        )

        return solution
