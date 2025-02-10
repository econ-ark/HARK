#! /usr/bin/env python3

"""Execute notebooks in parallel.

Reads the list of notebooks to execute from the command line.
If no notebooks are specified, all notebooks
in the examples directory are executed.

Notebooks are executed in parallel, with one worker
per processor in the host machine.
"""

import multiprocessing
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT_DIR = Path(__file__).resolve().parent.parent


def run_notebook(notebook_file: Path):
    rel_file_name = notebook_file.relative_to(ROOT_DIR).as_posix()
    print(f"{rel_file_name}: Loading notebook")
    try:
        # Journey-PhD and LifecycleModel expect execution from their own directory
        os.chdir(notebook_file.parent)
        nb = nbformat.read(notebook_file, as_version=4)
        client = NotebookClient(
            nb, timeout=600, kernel_name="python3", record_timing=False
        )
        print(f"{rel_file_name}: Executing")
        start = time.perf_counter()
        client.execute()
        elapsed = time.perf_counter() - start
        print(f"{rel_file_name}: Writing")
        nbformat.write(nb, notebook_file)
        print(f"{rel_file_name}: Finished (executed in {elapsed:.2f}s)")
        del nb, client, start, elapsed
    except Exception as err:
        print(f"{rel_file_name}: Failed to execute\n   {err}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        notebooks = (Path(p).resolve() for p in sys.argv[1:])
    else:
        notebooks = ROOT_DIR.joinpath("examples").rglob("*.ipynb")

    with multiprocessing.Pool() as pool:
        pool.map(run_notebook, notebooks)
