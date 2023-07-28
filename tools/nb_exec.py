#! /usr/bin/env python3

"""Execute notebooks in parallel.

Reads the list of notebooks to execute from the command line.
If no notebooks are specified, all notebooks
in the examples directory are executed.

Notebooks are executed in parallel, with one worker
per processor in the host machine.
"""

from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor

import nbformat
from nbclient import NotebookClient

ROOT_DIR = Path(__file__).parent.parent


def run_notebook(notebook_file):
    rel_file_name = notebook_file.relative_to(ROOT_DIR).as_posix()
    print(f'{rel_file_name}: Loading notebook')
    nb = nbformat.read(notebook_file, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    print(f'{rel_file_name}: Executing')
    client.execute()
    print(f'{rel_file_name}: Writing')
    nbformat.write(nb, notebook_file)
    del nb, client


if __name__ == '__main__':
    if len(sys.argv) > 1:
        notebooks = sys.argv[1:]
    else:
        notebooks = ROOT_DIR.joinpath('examples').rglob('*.ipynb')

    with ProcessPoolExecutor() as pool:
        pool.map(run_notebook, notebooks)
