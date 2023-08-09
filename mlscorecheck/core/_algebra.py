
"""
This module implements the joint interface to the algebraic systems
to be used.
"""

import importlib

__all__ = ['Algebra',
            'SympyAlgebra',
            'SageAlgebra',
            'Symbols']

class Algebra:
    """
    The base class of the algebra abstractions
    """
    def __init__(self):
        """
        The constructor of the algebra
        """
        pass

    def create_symbol(self, name, **kwargs):
        """
        Create a symbol in the algebra with the specified name and assumptions

        Args:
            name (str): the name of the symbol
            kwargs (dict): the assumptions

        Returns:
            object: the symbol
        """
        pass

    def num_denom(self, expression):
        """
        Extract the numerator and denominator

        Args:
            expression (object): the expression to process

        Returns:
            object, object: the numerator and denominator
        """
        pass

    def simplify(self, expression):
        """
        Simplify the expression

        Args:
            expression (object): the expression to simplify

        Returns:
            object: the symplified expression
        """
        pass

    def solve(self, equation, var):
        """
        Solve an equation for a variable

        Args:
            equation (object): the equation to solve
            var (object): the variable to solve for

        Returns:
            list(dict): the solutions
        """
        pass

    def subs(self, expression, subs_dict):
        """
        Substitute a substitution into the expression

        Args:
            expression (object): the expression to substitute into
            subs_dict (dict): the substitution

        Returns:
            object: the result of the substitution
        """
        pass

    def args(self, expression):
        """
        The list of arguments

        Args:
            expression (object): the expression to process

        Returns:
            list: the list of arguments
        """
        pass

    def is_trivial(self, expression):
        """
        Checks if the expression is trivial

        Args:
            expression (object): the expression to check

        Returns:
            bool: True if the expression is trivial, False otherwise
        """
        pass

    def is_root(self, expression):
        """
        Checks if the expression is a root

        Args:
            expression (object): the expression to check if it is a root

        Returns:
            bool: True if the expression is a root, False otherwise
        """
        pass

    def operands(self, expression):
        """
        Returns the list of operands

        Args:
            expression (object): the expression to return the operands of

        Returns:
            list: the operands
        """
        pass

    def free_symbols(self, expression):
        """
        Get all free symbols in an expression

        Args:
            expression (object): the expression to get the free symbols of

        Returns:
            list: the list of free symbols
        """
        pass

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

    def solve(self, equation, var):
        """
        Solve an equation for a variable

        Args:
            equation (object): the equation to solve
            var (object): the variable to solve for

        Returns:
            list(dict): the solutions
        """
        results = self.algebra.solve(equation, var)
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

        Args:
            expression (object): the expression to check

        Returns:
            bool: True if the expression is trivial, False otherwise
        """
        if expression is None:
            return True
        if expression == 1:
            return True

        return False

    def is_root(self, expression):
        """
        Checks if the expression is a root

        Args:
            expression (object): the expression to check if it is a root

        Returns:
            bool: True if the expression is a root, False otherwise
        """
        if isinstance(expression, self.algebra.core.power.Pow):
            base, exponent = expression.args()
            if exponent < 1:
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
    The required algebra driven by sympy
    """
    def __init__(self):
        """
        Constructor of the algebra
        """
        Algebra.__init__(self)

        self.algebra = importlib.import_module("sage").all
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

    def solve(self, equation, var):
        """
        Solve an equation for a variable

        Args:
            equation (object): the equation to solve
            var (object): the variable to solve for

        Returns:
            list(dict): the solutions
        """
        results = self.algebra.solve(equation, var)
        solutions = []
        for sol in results:
            solution = {}
            solution[sol.lhs()] = self.algebra.factor(sol.rhs())
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
        return expression.args()

    def is_trivial(self, expression):
        """
        Checks if the expression is trivial

        Args:
            expression (object): the expression to check

        Returns:
            bool: True if the expression is trivial, False otherwise
        """
        if expression is None:
            return True
        if expression.is_trivially_equal(1):
            return True
        return False

    def is_root(self, expression):
        """
        Checks if the expression is a root

        Args:
            expression (object): the expression to check if it is a root

        Returns:
            bool: True if the expression is a root, False otherwise
        """
        if hasattr(expression.operator(), '__qualname__') and expression.operator().__qualname__ == 'pow':
            base, exponent = expression.operands()
            if exponent < 1:
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
        return expression.operands()

    def free_symbols(self, expression):
        """
        Get all free symbols in an expression

        Args:
            expression (object): the expression to get the free symbols of

        Returns:
            list: the list of free symbols
        """
        return [str(var) for var in list(expression.free_variables())]

class Symbols:
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

