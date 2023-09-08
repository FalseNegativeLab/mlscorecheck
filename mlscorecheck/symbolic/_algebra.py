
"""
This module implements the joint interface to the algebraic systems
to be used.
"""

import abc

import importlib

__all__ = ['Algebra',
            'SympyAlgebra',
            'SageAlgebra',
            'Symbols']

class Algebra(metaclass=abc.ABCMeta):
    """
    The base class of the algebra abstractions
    """
    @abc.abstractmethod
    def __init__(self):
        """
        The constructor of the algebra
        """

    @abc.abstractmethod
    def create_symbol(self, name, **kwargs):
        """
        Create a symbol in the algebra with the specified name and assumptions

        Args:
            name (str): the name of the symbol
            kwargs (dict): the assumptions

        Returns:
            object: the symbol
        """

    @abc.abstractmethod
    def num_denom(self, expression):
        """
        Extract the numerator and denominator

        Args:
            expression (object): the expression to process

        Returns:
            object, object: the numerator and denominator
        """

    @abc.abstractmethod
    def simplify(self, expression):
        """
        Simplify the expression

        Args:
            expression (object): the expression to simplify

        Returns:
            object: the symplified expression
        """

    @abc.abstractmethod
    def solve(self, equation, var):
        """
        Solve an equation for a variable

        Args:
            equation (object): the equation to solve
            var (object): the variable to solve for

        Returns:
            list(dict): the solutions
        """

    @abc.abstractmethod
    def subs(self, expression, subs_dict):
        """
        Substitute a substitution into the expression

        Args:
            expression (object): the expression to substitute into
            subs_dict (dict): the substitution

        Returns:
            object: the result of the substitution
        """

    @abc.abstractmethod
    def args(self, expression):
        """
        The list of arguments

        Args:
            expression (object): the expression to process

        Returns:
            list: the list of arguments
        """

    @abc.abstractmethod
    def is_trivial(self, expression):
        """
        Checks if the expression is trivial

        Args:
            expression (object): the expression to check

        Returns:
            bool: True if the expression is trivial, False otherwise
        """

    @abc.abstractmethod
    def is_root(self, expression):
        """
        Checks if the expression is a root

        Args:
            expression (object): the expression to check if it is a root

        Returns:
            bool: True if the expression is a root, False otherwise
        """

    @abc.abstractmethod
    def is_division(self, expression):
        """
        Checks if the expression is a division

        Args:
            expression (object): the expression to check if it is a division

        Returns:
            bool: True if the expression is a division, False otherwise
        """

    @abc.abstractmethod
    def is_power(self, expression):
        """
        Checks if the expression is a power

        Args:
            expression (object): the expression to check if it is a power

        Returns:
            bool: True if the expression is a power, False otherwise
        """

    @abc.abstractmethod
    def operands(self, expression):
        """
        Returns the list of operands

        Args:
            expression (object): the expression to return the operands of

        Returns:
            list: the operands
        """

    @abc.abstractmethod
    def free_symbols(self, expression):
        """
        Get all free symbols in an expression

        Args:
            expression (object): the expression to get the free symbols of

        Returns:
            list: the list of free symbols
        """

class SympyAlgebra(Algebra):
    """
    The required algebra driven by sympy
    """
    def __init__(self):
        """
        Constructor of the algebra
        """
        Algebra.__init__(self)

        self.algebra = importlib.import_module("sympy")
        self.sqrt = self.algebra.sqrt

    def create_symbol(self, name, **kwargs):
        """
        Create a symbol in the algebra with the specified name and assumptions

        Args:
            name (str): the name of the symbol
            kwargs (dict): the assumptions

        Returns:
            object: the symbol
        """
        if 'upper_bound' in kwargs:
            del kwargs['upper_bound']
        if 'lower_bound' in kwargs:
            del kwargs['lower_bound']
        return self.algebra.Symbol(name, **kwargs)

    def num_denom(self, expression):
        """
        Extract the numerator and denominator

        Args:
            expression (object): the expression to process

        Returns:
            object, object: the numerator and denominator
        """
        return expression.as_numer_denom()

    def simplify(self, expression):
        """
        Simplify the expression

        Args:
            expression (object): the expression to simplify

        Returns:
            object: the symplified expression
        """
        return self.algebra.simplify(expression)

    def solve(self, equation, var, **kwargs):
        """
        Solve an equation for a variable

        Args:
            equation (object): the equation to solve
            var (object): the variable to solve for
            kwargs (dict): additional parameters to the solver

        Returns:
            list(dict): the solutions
        """
        results = self.algebra.solve(equation, var, **kwargs)
        solutions = []
        for res in results:
            solutions.append({var: res})
        return solutions

    def subs(self, expression, subs_dict):
        """
        Substitute a substitution into the expression

        Args:
            expression (object): the expression to substitute into
            subs_dict (dict): the substitution

        Returns:
            object: the result of the substitution
        """
        return expression.subs(subs_dict)

    def args(self, expression):
        """
        The list of arguments

        Args:
            expression (object): the expression to process

        Returns:
            list: the list of arguments
        """
        return expression.free_symbols

    def is_trivial(self, expression):
        """
        Checks if the expression is trivial
        TODO: checking other constants

        Args:
            expression (object): the expression to check

        Returns:
            bool: True if the expression is trivial, False otherwise
        """
        return True if expression is None else expression == 1

    def is_root(self, expression):
        """
        Checks if the expression is a root

        Args:
            expression (object): the expression to check if it is a root

        Returns:
            bool: True if the expression is a root, False otherwise
        """
        if self.is_power(expression):
            _, exponent = expression.args
            if 0 < exponent < 1:
                return True
        return False

    def is_power(self, expression):
        """
        Checks whether the expression is a power

        Args:
            expression (object): the expression to check

        Returns:
            bool: whether the expression is a power
        """
        return isinstance(expression, self.algebra.core.power.Pow)

    def is_division(self, expression):
        """
        Checks whether the expression is a division

        Args:
            expression (object): the expression to check

        Returns:
            bool: whether the expression is a division
        """
        if self.is_power(expression):
            _, power = expression.args
            if power < 0:
                return True

        if isinstance(expression, self.algebra.core.power.Mul):
            args = expression.args
            if len(args) == 2 and self.is_power(args[1]):
                _, power = args[1].args
                if power < 0:
                    return True
        return False

    def operands(self, expression):
        """
        Returns the list of operands

        Args:
            expression (object): the expression to return the operands of

        Returns:
            list: the operands
        """
        return expression.args

    def free_symbols(self, expression):
        """
        Get all free symbols in an expression

        Args:
            expression (object): the expression to get the free symbols of

        Returns:
            list: the list of free symbols
        """
        return [str(var) for var in list(expression.free_symbols)]

class SageAlgebra(Algebra):
    """
    The required algebra driven by sage
    """
    def __init__(self):
        """
        Constructor of the algebra
        """
        Algebra.__init__(self)

        self.algebra = importlib.import_module("sage.all")
        self.sqrt = self.algebra.sqrt

    def create_symbol(self, name, **kwargs):
        """
        Create a symbol in the algebra with the specified name and assumptions

        Args:
            name (str): the name of the symbol
            kwargs (dict): the assumptions

        Returns:
            object: the symbol
        """
        var = self.algebra.var(name)
        if kwargs.get('nonnegative', False):
            self.algebra.assume(var >= 0)
        if kwargs.get('positive', False):
            self.algebra.assume(var > 0)
        if kwargs.get('negative', False):
            self.algebra.assume(var < 0)
        if kwargs.get('nonpositive', False):
            self.algebra.assume(var <= 0)
        if kwargs.get('real', False):
            self.algebra.assume(var, 'real')
        if kwargs.get('upper_bound', None) is not None:
            self.algebra.assume(var <= kwargs['upper_bound'])
        if kwargs.get('lower_bound', None) is not None:
            self.algebra.assume(var >= kwargs['lower_bound'])

        return var

    def num_denom(self, expression):
        """
        Extract the numerator and denominator

        Args:
            expression (object): the expression to process

        Returns:
            object, object: the numerator and denominator
        """
        return expression.numerator(), expression.denominator()

    def simplify(self, expression):
        """
        Simplify the expression

        Args:
            expression (object): the expression to simplify

        Returns:
            object: the symplified expression
        """
        return self.algebra.factor(expression)

    def solve(self, equation, var, **kwargs):
        """
        Solve an equation for a variable

        Args:
            equation (object): the equation to solve
            var (object): the variable to solve for
            kwargs (dict): additional parameters to the solver

        Returns:
            list(dict): the solutions
        """
        results = self.algebra.solve(equation, var, **kwargs)
        solutions = []
        for sol in results:
            solution = {sol.lhs(): self.algebra.factor(sol.rhs())}
            solutions.append(solution)
        return solutions

    def subs(self, expression, subs_dict):
        """
        Substitute a substitution into the expression

        Args:
            expression (object): the expression to substitute into
            subs_dict (dict): the substitution

        Returns:
            object: the result of the substitution
        """
        return expression.subs(subs_dict)

    def args(self, expression):
        """
        The list of arguments

        Args:
            expression (object): the expression to process

        Returns:
            list: the list of arguments
        """
        return set(expression.args())

    def is_trivial(self, expression):
        """
        Checks if the expression is trivial

        Args:
            expression (object): the expression to check

        Returns:
            bool: True if the expression is trivial, False otherwise
        """
        return True if expression is None else expression.is_trivially_equal(1)

    def is_root(self, expression):
        """
        Checks if the expression is a root

        Args:
            expression (object): the expression to check if it is a root

        Returns:
            bool: True if the expression is a root, False otherwise
        """
        if self.is_power(expression):
            _, exponent = expression.operands()
            if 0 < exponent < 1:
                return True
        return False

    def is_power(self, expression):
        """
        Checks whether the expression is a power

        Args:
            expression (object): the expression to check

        Returns:
            bool: whether the expression is a power
        """
        return bool(hasattr(expression.operator(), '__qualname__')\
                    and expression.operator().__qualname__ == 'pow')

    def is_division(self, expression):
        """
        Checks whether the expression is a division

        Args:
            expression (object): the expression to check

        Returns:
            bool: whether the expression is a division
        """
        if self.is_power(expression):
            operands = expression.operands()
            if operands[1] < 0:
                return True

        if hasattr(expression.operator(), '__qualname__') \
            and expression.operator().__qualname__ == 'mul_vararg':
            operands = expression.operands()

            if len(operands) == 2:
                if self.is_power(operands[1]):
                    _, power = operands[1].operands()
                    if power < 0:
                        return True
                elif self.is_power(operands[0]):
                    _, power = operands[0].operands()
                    if power < 0:
                        return True
        return False

    def operands(self, expression):
        """
        Returns the list of operands

        Args:
            expression (object): the expression to return the operands of

        Returns:
            list: the operands
        """
        return tuple(expression.operands())

    def free_symbols(self, expression):
        """
        Get all free symbols in an expression

        Args:
            expression (object): the expression to get the free symbols of

        Returns:
            list: the list of free symbols
        """
        return [str(var) for var in list(expression.free_variables())]

class Symbols: # pylint: disable=too-many-instance-attributes
    """
    A symbols class representing the basic symbols to be used
    """
    def __init__(self, algebraic_system):
        """
        The constructor of the object

        Args:
            algebraic_system ('sympy'/'sage'): the algebraic system to be used
        """
        self.algebraic_system = algebraic_system
        if algebraic_system == 'sympy':
            self.algebra = SympyAlgebra()
        elif algebraic_system == 'sage':
            self.algebra = SageAlgebra()

        self.tp = self.algebra.create_symbol('tp', nonnegative=True, real=True)
        self.tn = self.algebra.create_symbol('tn', nonnegative=True, real=True)
        self.p = self.algebra.create_symbol('p', positive=True, real=True)
        self.n = self.algebra.create_symbol('n', positive=True, real=True)
        self.beta_plus = self.algebra.create_symbol('beta_plus', positive=True, real=True)
        self.beta_minus = self.algebra.create_symbol('beta_minus', positive=True, real=True)
        self.sqrt = self.algebra.sqrt

    def get_algebra(self):
        """
        Returns the algebra

        Returns:
            Algebra: the algebra object
        """
        return self.algebra

    def to_dict(self):
        """
        Returns a dictionary representation

        Returns:
            dict: the dictionary representation
        """
        return {'tp': self.tp,
                'tn': self.tn,
                'p': self.p,
                'n': self.n,
                'beta_plus': self.beta_plus,
                'beta_minus': self.beta_minus,
                'sqrt': self.sqrt}
