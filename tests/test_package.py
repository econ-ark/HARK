from __future__ import annotations

import importlib.metadata

import HARK as m


def test_version():
    assert importlib.metadata.version("HARK") == m.__version__
