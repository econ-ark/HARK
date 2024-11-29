"""
Copyright (c) 2024 Econ-ARK Team. All rights reserved.

HARK: Heterogeneous Agents Resources and toolKit
"""

from __future__ import annotations

from ._version import version as __version__

__all__ = ["__version__"]

from .core import *


"""
Logging tools for HARK.

The logger will print logged statements to STDOUT by default.

The logger wil use an informative value by default.
The user can set it to "verbose" to get more information, or "quiet" to supress informative messages.
"""

import logging

logging.basicConfig(format="%(message)s")

_log = logging.getLogger("HARK")

_log.setLevel(logging.ERROR)


def disable_logging():
    _log.disabled = True


def enable_logging():
    _log.disabled = False


def warnings():
    _log.setLevel(logging.WARNING)


def quiet():
    _log.setLevel(logging.ERROR)


def verbose():
    _log.setLevel(logging.INFO)


def set_verbosity_level(level):
    _log.setLevel(level)
