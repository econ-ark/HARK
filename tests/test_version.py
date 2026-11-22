"""HARK's version comes from git, through setuptools_scm (see pyproject.toml)."""

from packaging.version import Version

import HARK


def test_version_is_pep440():
    # A release's version is its tag; any other commit gets a development
    # version such as 0.18.0.dev270+g7571205e. pip compares versions under
    # PEP 440, so the version must parse as one.
    Version(HARK.__version__)
