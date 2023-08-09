"""
This module implements the logger
"""

import logging

__all__ = ['logger']

logger = logging.getLogger("mlscorecheck")

# setting the _logger format
logger.setLevel(logging.DEBUG)
logger_ch = logging.StreamHandler()
logger_ch.setFormatter(logging.Formatter(
    "%(asctime)s:%(levelname)s:%(message)s"))
logger.addHandler(logger_ch)
