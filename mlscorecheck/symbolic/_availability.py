"""
This module implements a function to check if a symbolic toolkit is available
"""

__all__ = ['symbolic_toolkits',
            'get_symbolic_toolkit']

symbolic_toolkits = []

try:
    import sympy
    symbolic_toolkits.append('sympy')
except ImportError as exception:
    pass

try:
    import sage
    symbolic_toolkits.append('sage')
except ImportError as exception:
    pass

def get_symbolic_toolkit():
    if len(symbolic_toolkits) > 0:
        return symbolic_toolkits[0]
    return None
