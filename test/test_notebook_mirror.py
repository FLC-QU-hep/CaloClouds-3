"""
Notebooks should be converted to python files before being commited to the repository.
"""
from programming_utils import mirror_notebook
import os

_generic_hooks_notice = (
    "If this is failing, you might not have the pre-commit hooks installed. "
    + "Review the hooks in .githooks to check you are happy with them, "
    + "then install them with `make enable-git-hooks`"
)


def assert_has_mirror(notebook_path):
    """
    For a specific notebook, check that the mirror file exists and
    contains the same content as the notebook.
    """
    mirror_file = mirror_notebook.get_mirror_path(notebook_path)
    assert os.path.isfile(
        mirror_file
    ), f"Notebook {notebook_path} does not have a mirror file {mirror_file}. {_generic_hooks_notice}"
    # check the content
    expected_mirror = mirror_notebook.extract(notebook_path)
    with open(mirror_file, "r") as f:
        actual_mirror = f.read()
    assert (
        expected_mirror.strip() == actual_mirror.strip()
    ), f"Mirror file {mirror_file} does not match the notebook {notebook_path}. {_generic_hooks_notice}"


def test_check_notebooks():
    notebooks = mirror_notebook.list_of_notebooks()
    for notebook in notebooks:
        assert_has_mirror(notebook)
