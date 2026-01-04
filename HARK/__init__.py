"""
Logging tools for HARK.

The logger will print logged statements to STDOUT by default.

The logger wil use an informative value by default.
The user can set it to "verbose" to get more information, or "quiet" to supress informative messages.
"""

__all__ = [
    "AgentType",
    "Market",
    "Parameters",
    "Model",
    "AgentPopulation",
    "multi_thread_commands",
    "multi_thread_commands_fake",
    "NullFunc",
    "make_one_period_oo_solver",
    "distribute_params",
    "install_examples",
]


from .core import *

__version__ = "0.17.0"
import logging
from HARK.helpers import install_examples

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
