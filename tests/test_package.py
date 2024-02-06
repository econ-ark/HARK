from __future__ import annotations

import importlib.metadata

import hark as m


def test_version():
    assert importlib.metadata.version("hark") == m.__version__
