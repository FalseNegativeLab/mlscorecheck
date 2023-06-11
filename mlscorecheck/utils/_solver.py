"""
This module implements the sympy based problem solver
"""

import numpy as np

import sympy
from sympy import Symbol

from ._problem_generator import generate_problem
from ..core import Interval, union

import logging

__all__ = ['ScoreFunction',
           'ProblemSolver']

# configuring the logger
logging.basicConfig(
format='%(asctime)s %(levelname)-8s %(message)s',
level=logging.INFO,
datefmt='%Y-%m-%d %H:%M:%S')

class ScoreFunction:
    """
    A simple data class representing a score function and its symbolic version
    """
    def __init__(self, name, function, atomic_symbols):
        """
        Constructor of the object

        Args:
            name (str): the score
            function (callable): the score function
            atomic_symbols (dict): the atomic symbols
        """
        self.function = function
        self.name = name
        self.score_sym = Symbol(self.name)
        self.args = function.__code__.co_varnames[:function.__code__.co_kwonlyargcount]

        tmp_dict = {key: atomic_symbols[key] for key in self.args}

        self.function_sym = function(**tmp_dict)
        self.equation = self.score_sym - self.function_sym

class ProblemSolver:
    """
    Solves tp and tn for two score functions
    """
    def __init__(self,
                    name0,
                    function0,
                    name1,
                    function1,
                    tol=1e-8):
        """
        Constructor of the object

        Args:
            name0 (str): the name of the first score
            function0 (callable): the first score function
            name1 (str): the name of the second score
            function1 (callable): the second score function
            tol (float, optional): the tolerance for checking the validity of the solutions.
                                    Defaults to 1e-8.
        """
        self.tol = tol

        self.atomic_symbols = {'tp': Symbol('tp'),
                                'fp': Symbol('fp'),
                                'tn': Symbol('tn'),
                                'fn': Symbol('fn'),
                                'p': Symbol('p'),
                                'n': Symbol('n'),
                                'beta': Symbol('beta')}

        self.score0 = ScoreFunction(name0, function0, self.atomic_symbols)
        self.score1 = ScoreFunction(name1, function1, self.atomic_symbols)

        self.equation_p = self.atomic_symbols['p'] - self.atomic_symbols['tp'] - self.atomic_symbols['fn']
        self.equation_n = self.atomic_symbols['n'] - self.atomic_symbols['tn'] - self.atomic_symbols['fp']

        self.symbols_to_solve = [self.atomic_symbols['tp'], self.atomic_symbols['tn'],
                                    self.atomic_symbols['fp'], self.atomic_symbols['fn']]

        self.solutions = None

        self.linear = False

    def names(self):
        if self.solutions is not None:
            tmp = []
            if len(self.solutions) > 1:
                tmp.extend(f'_solve_{self.score0.name}_{self.score1.name}_{idx}'
                                                for idx in range(len(self.solutions)))
            tmp.append(f'solve_{self.score0.name}_{self.score1.name}')
        return tmp

    def solve(self):
        """
        Solves the problem and sets the flags accordingly
        """
        eqs = [self.score0.equation,
                self.score1.equation,
                self.equation_p,
                self.equation_n]

        results = sympy.solve(eqs,
                                self.symbols_to_solve,
                                dict=True,
                                check=False)

        logging.info(f'number of non-linear solutions {len(results)}')

        logging.info(f'res {results}')

        self.solutions = []
        for result in results:
            sol = {}
            if self.atomic_symbols['tp'] in result:
                sol['tp'] = result[self.atomic_symbols['tp']]

            if self.atomic_symbols['tn'] in result:
                sol['tn'] = result[self.atomic_symbols['tn']]

            self.solutions.append(sol)

        try:
            results = sympy.linsolve([self.score0.equation,
                                            self.score1.equation,
                                            self.equation_p,
                                            self.equation_n],
                                            self.symbols_to_solve)

            logging.info(f'number of linear solutions {len(results)}')

            if len(results) > 0:
                self.linear = True

        except Exception as exc:
            logging.info(f"Linear solution failed {exc}")

    def remove_zoo_solutions(self):
        """
        Removing the solutions containing complex infinity (zoo in sympy)
        """
        zoo = sympy.core.numbers.ComplexInfinity()
        self.new_solutions = [sol for sol in self.solutions
                            if zoo not in sol['tp'].atoms()
                            and zoo not in sol['tn'].atoms()]
        n_removed = len(self.solutions) - len(self.new_solutions)
        if n_removed > 0:
            logging.info(f'{self.score0.name} - {self.score1.name}: number of '\
                                            f'zoo solutions removed: {n_removed}')
        self.solutions = self.new_solutions

        return n_removed

    def all_free_symbols(self):
        free_symbols = set()

        for idx in range(len(self.solutions)):
            symb_idx = (self.solutions[idx]['tp'].free_symbols | self.solutions[idx]['tn'].free_symbols)
            free_symbols = free_symbols | symb_idx

        return {str(symb) for symb in free_symbols}

    def degenerate_problem(self):
        symbols = self.all_free_symbols()
        if self.score0.name not in symbols or self.score1.name not in symbols:
            logging.info(f'{self.score0.name} - {self.score1.name}: degenerate problem')
            return True
        return False

    def solution_to_python(self, sol_idx):
        score0 = self.score0.name
        score1 = self.score1.name
        tp_sol = str(self.solutions[sol_idx]['tp'])
        tn_sol = str(self.solutions[sol_idx]['tn'])

        free_symbols = (self.solutions[sol_idx]['tp'].free_symbols | self.solutions[sol_idx]['tn'].free_symbols)
        if len(self.solutions) == 1:
            free_symbols = {str(symb) for symb in free_symbols}.union({'p', 'n'})
        else:
            free_symbols = {str(symb) for symb in free_symbols}

        final_symbols = [symb for symb in [score0, score1, 'p', 'n', 'beta'] if symb in free_symbols]

        free_symbol_spec = "".join([f"    {score} (int/float/Interval): the {score} value\n"
                                    for score in final_symbols[2:]])

        doc_str =\
        f'  """\n'\
        f"  Solution of for the {score0} - {score1} pair\n\n"\
        f"  Args:\n"\
        f"    {score0} (int/float/Interval): the {score0} score\n"\
        f"    {score1} (int/float/Interval): the {score1} score\n"\
        f"{free_symbol_spec}\n"\
        f"  Returns:\n"\
        f"    float/Interval, float/Interval: the tp and tn solutions\n"\
        f'  """\n'

        func_name = None
        if len(self.solutions) == 1:
            func_name = f"solve_{score0}_{score1}"
        else:
            func_name = f"_solve_{score0}_{score1}_{sol_idx}"

        function_str = \
        f"def {func_name}(*, {', '.join(final_symbols)}):\n"\
        f"{doc_str}"\
        f"  try:\n"\
        f"    tp = {tp_sol}\n"\
        f"    tn = {tn_sol}\n"\
        f"  except ZeroDivisionError:\n"\
        f"    tp = np.nan\n"\
        f"    tn = np.nan\n"

        if len(self.solutions) > 1:
            function_str += f"  return tp, tn\n\n"
        if len(self.solutions) == 1:
            function_str += f"  return reduce_solutions([tp], [tn], p, n)\n\n"

        return function_str

    def wrapper_to_python(self):

        score0 = self.score0.name
        score1 = self.score1.name

        free_symbols = set()

        for idx in range(len(self.solutions)):
            symb_idx = (self.solutions[idx]['tp'].free_symbols | self.solutions[idx]['tn'].free_symbols)
            free_symbols = free_symbols | symb_idx
        free_symbols = {str(symb) for symb in free_symbols}.union({'p', 'n'})

        if len(self.solutions) > 0:
            if 'p' not in free_symbols:
                free_symbols.add('p')
            if 'n' not in free_symbols:
                free_symbols.add('n')

        final_symbols = [symb for symb in [score0, score1, 'p', 'n', 'beta'] if symb in free_symbols]

        free_symbol_spec = "".join([f"    {score} (int/float/Interval): the {score} value\n"
                                    for score in final_symbols[2:]])

        doc_str =\
        f'  """\n'\
        f"  Solution of for the {score0} - {score1} pair\n\n"\
        f"  Args:\n"\
        f"    {score0} (int/float/Interval): the {score0} score\n"\
        f"    {score1} (int/float/Interval): the {score1} score\n"\
        f"{free_symbol_spec}\n"\
        f"  Returns:\n"\
        f"    float/Interval, float/Interval: the tp and tn solutions\n"\
        f'  """\n'

        #argscall = [f'{sym}={sym}' for sym in final_symbols]

        calls = ""
        for sol_idx in range(len(self.solutions)):
            free_symbols = (self.solutions[sol_idx]['tp'].free_symbols | self.solutions[sol_idx]['tn'].free_symbols)
            free_symbols = {str(symb) for symb in free_symbols}

            #final_symbols = [score0, score1] + [symb for symb in ['p', 'n', 'beta'] if symb in free_symbols]
            final_symbols_call = [symb for symb in [score0, score1, 'p', 'n', 'beta'] if symb in free_symbols]
            argscall = [f'{sym}={sym}' for sym in final_symbols_call]

            calls += f"  tp, tn = _solve_{score0}_{score1}_{sol_idx}({', '.join(argscall)})\n"
            calls += "  tps.append(tp)\n"
            calls += "  tns.append(tn)\n"

        return \
        f"def solve_{score0}_{score1}(*, {', '.join(final_symbols)}):\n"\
        f"{doc_str}"\
        f"  tps = []\n"\
        f"  tns = []\n"\
        f"{calls}"\
        f"  return reduce_solutions(tps, tns, p, n)\n\n"

    def to_python(self):
        if len(self.solutions) > 1:
            codes = "".join([self.solution_to_python(idx) for idx in range(len(self.solutions))])
            codes += self.wrapper_to_python()
            return codes
        return self.solution_to_python(0)

    def __str__(self):
        """
        Generate a Python-function from the solution
        """
        score0 = self.score0.name
        score1 = self.score1.name
        tp_sol = str(self.tp)
        tn_sol = str(self.tn)

        free_symbols = (self.tp.free_symbols | self.tn.free_symbols)
        free_symbols = {str(symb) for symb in free_symbols}

        final_symbols = [score0, score1] + [symb for symb in ['p', 'n', 'beta'] if symb in free_symbols]

        free_symbol_spec = "".join([f"    {score} (int/float/Interval): the {score} value\n"
                                    for score in final_symbols[2:]])

        doc_str =\
        f'  """\n'\
        f"  Solution of for the {score0} - {score1} pair\n\n"\
        f"  Args:\n"\
        f"    {score0} (int/float/Interval): the {score0} score\n"\
        f"    {score1} (int/float/Interval): the {score1} score\n"\
        f"{free_symbol_spec}\n"\
        f"  Returns:\n"\
        f"    float/Interval, float/Interval: the tp and tn solutions\n"\
        f'  """\n'

        return \
        f"def solve_{score0}_{score1}(*, {', '.join(final_symbols)}):\n"\
        f"{doc_str}"\
        f"  tp = {tp_sol}\n"\
        f"  tn = {tn_sol}\n"\
        f"  return tp, tn\n\n"
