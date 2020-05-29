from .core import *

__version__ = "0.10.6"

'''
Logging tools for HARK.

The logger will print logged statements to STDOUT by default.

The logger wil use an informative value by default.
The user can set it to "verbose" to get more information, or "quiet" to supress informative messages.
'''

import logging

logging.basicConfig(
    format="%(message)s"
)

_log = logging.getLogger("HARK")

_log.setLevel(logging.INFO)

def verbose():
    _log.setLevel(logging.DEBUG)
    
def quiet():
    _log.setLevel(logging.WARNING)
