


import glob
import os.path
import sys
from concurrent.futures import ProcessPoolExecutor

import nbformat
from nbclient import NotebookClient


def run_notebook(notebook_file):
    print(f'Loading {notebook_file}')
    nb = nbformat.read(notebook_file, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    print(f'Executing {notebook_file}')
    client.execute()
    print(f'Writing {notebook_file}')
    nbformat.write(nb, notebook_file)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        notebooks = sys.argv[1:]
    else:
        notebooks = glob.glob('**/*.ipynb', root_dir='../examples', recursive=True)

    with ProcessPoolExecutor() as pool:
        pool.map(run_notebook, notebooks)
