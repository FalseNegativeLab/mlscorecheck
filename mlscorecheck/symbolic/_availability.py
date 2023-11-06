"""
This module implements a function to check if a symbolic toolkit is available
"""

import importlib

__all__ = ["symbolic_toolkits", "get_symbolic_toolkit", "check_importability"]

symbolic_toolkits = []


def check_importability(package: str):
    """
    Tests the importability of a package

    Args:
        str: the name of the package

    Returns:
        str|None: the name of the package if importable or None if not importable
    """
    try:
        _ = importlib.import_module(package)
        return package
    except ModuleNotFoundError:
        return None


symbolic_toolkits.append(check_importability("sympy"))
symbolic_toolkits.append(check_importability("sage"))

symbolic_toolkits = [package for package in symbolic_toolkits if package is not None]


def get_symbolic_toolkit() -> str:
    """
    Returns the name of an available symbolic toolkit (in sympy, sage order)

    Returns:
        str: the name of the available module
    """
    return symbolic_toolkits[0] if len(symbolic_toolkits) > 0 else None
