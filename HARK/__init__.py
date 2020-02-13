from __future__ import absolute_import
import sys

if sys.version_info[:2] < (3, 6):
    print(
        "WARNING: Econ-Ark v0.10.4 will be the last supported version for Python %d.%d, "
        "please upgrade your system to the latest release, read more at https://www.python.org/doc/sunset-python-2/"
        % sys.version_info[:2]
    )


from .core import *

__version__ = "0.10.3"
