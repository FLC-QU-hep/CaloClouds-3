#!/usr/bin/env python
"""
Module to make mirrored copies of pynb/ipynb files as regular python files.


Works on the basis that the notebook is a json file, and that the code cells
are stored in the 'cells' key of the json object.
Ignores images, and prints markdown cells as comments.

If it's called as a script, it will mirror all the notebooks in the repository.
"""
from pathlib import Path
import json
import sys
import os


def extract(notebook_path):
    """
    Make a mirrored copy of a pynb/ipynb file as a regular python file.

    Parameters
    ----------
    notebook_path : str or Path
        Path to the notebook file.

    Returns
    -------
    extracted : str
        The extracted code from the notebook.
    """
    with open(notebook_path, "r") as f:
        notebook = json.load(f)
    extracted = ""
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            extracted += "".join(cell["source"]) + "\n"
        elif cell["cell_type"] == "markdown":
            extracted += "".join("# " + line for line in cell["source"]) + "\n"
    return extracted


def get_mirror_path(notebook_path):
    """
    Get the path of the mirrored python file for a given pynb/ipynb file.

    Parameters
    ----------
    notebook_path : str or Path
        Path to the notebook file.

    Returns
    -------
    mirror_path : str
        Path to the mirrored python file.
    """
    if isinstance(notebook_path, Path):
        notebook_path = str(notebook_path)
    return notebook_path + "_mirror.py"


def mirror(notebook_path, python_path=None):
    """
    Write a mirrored copy of a pynb/ipynb file as a regular python file.

    Parameters
    ----------
    notebook_path : str or Path
        Path to the notebook file.
    python_path : str or Path, optional
        Path to the mirrored python file. If not given, the path is
        notebook_path + '_mirror.py'.

    Returns
    -------
    python_path : str
        Path to the mirrored python file.
    """
    if python_path is None:
        python_path = get_mirror_path(notebook_path)
    with open(python_path, "w") as f:
        f.write(extract(notebook_path))
    return python_path


def list_of_notebooks():
    """
    Get a list of all the notebooks in the repository.

    Returns
    -------
    notebooks : list of Path
        List of paths to the notebooks.
    """
    path_root = Path(__file__).parents[1]
    # iterate over all the subfolders looking for notebooks
    notebooks = []
    for root, dirs, files in os.walk(path_root):
        for file in files:
            if "/.ipynb_checkpoints" in root:
                continue  # ignore checkpoints
            if file.endswith("pynb"):
                notebooks.append(Path(root) / file)
    return notebooks


def main():
    """
    Main function to mirror all notebooks in the repository.
    """
    notebooks = list_of_notebooks()
    for notebook in notebooks:
        python_path = mirror(notebook)
        print(python_path)


if __name__ == "__main__":
    main()
