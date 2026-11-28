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

# The version is set from git when HARK is built or installed (setuptools_scm,
# configured in pyproject.toml), which writes HARK/_version.py.
try:
    from HARK._version import version as __version__
except ImportError:  # a tree that was never built or installed
    from importlib.metadata import PackageNotFoundError, distribution
    from pathlib import Path as _Path

    # The installed econ-ark's metadata gives the version only if that
    # installation is the tree being imported, not, say, an older install
    # beside a source tree on PYTHONPATH
    __version__ = "0+unknown"
    try:
        _dist = distribution("econ-ark")
        if (
            _Path(_dist.locate_file("HARK/__init__.py")).resolve()
            == _Path(__file__).resolve()
        ):
            __version__ = _dist.version
    except PackageNotFoundError:
        pass
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
