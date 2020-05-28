'''
Logging tools for HARK.

The logger will print logged statements to STDOUT by default.

The logger wil use an informative value by default.
The user can set it to "verbose" to get more information, or "quiet" to supress informative messages.
'''

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)

def verbose():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s"
    )
    
def quiet():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s"
    )
