"""HARK's version comes from git, through setuptools_scm (see pyproject.toml)."""

import fnmatch
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
from packaging.version import Version

import HARK

ROOT = Path(__file__).resolve().parents[1]
SCM = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools_scm"]


def describe_filters():
    """The --match and --exclude patterns of pyproject.toml's describe_command."""
    command = SCM["scm"]["git"]["describe_command"]
    pairs = list(zip(command, command[1:]))
    match = [pattern for option, pattern in pairs if option == "--match"]
    exclude = [pattern for option, pattern in pairs if option == "--exclude"]
    return match, exclude


def test_version_is_pep440():
    # A release's version is its tag; any other commit gets a development
    # version such as 0.18.0.dev12+g7571205e. pip compares versions under
    # PEP 440, so the version must parse as one.
    Version(HARK.__version__)


def test_the_tag_filters_agree():
    # git describe takes the nearest tag that describe_command's patterns
    # admit, and setuptools_scm reads it with tag_regex, so a tag the patterns
    # admit but tag_regex rejects would be found and then fail. The archive's
    # placeholder in .git_archival.txt, for installs from a GitHub archive,
    # takes the same patterns.
    match, exclude = describe_filters()
    accepted = [
        "0.17.2",
        "0.18",
        "0.18.0",
        "0.18.0rc1",
        "0.18rc1",
        "0.18.0a2",
        "0.18.0b3",
        "0.18.0.dev0",
        "0.18.0rc1.dev0",
        "10.20.30",
    ]
    rejected = [
        "0.16.1b",
        "0.10.0.dev1",
        "0.18.0.dev10",
        "0.18.0.dev01",
        "0.17.2.1",
        "0.17.0.post1",
        "0.18.0-rc1",
        "v0.17.0.post1-broadcasting-fix",
        "hafiscal/2026-10-20.1",
        "latest",
    ]
    for tag in accepted + rejected:
        by_regex = re.match(SCM["tag_regex"], tag) is not None
        by_patterns = any(fnmatch.fnmatchcase(tag, p) for p in match) and not any(
            fnmatch.fnmatchcase(tag, p) for p in exclude
        )
        assert by_regex == by_patterns == (tag in accepted), tag

    archival = (ROOT / ".git_archival.txt").read_text()
    options = re.search(r"%\(describe:([^)]*)\)", archival).group(1).split(",")
    assert [o[len("match=") :] for o in options if o.startswith("match=")] == match
    assert [
        o[len("exclude=") :] for o in options if o.startswith("exclude=")
    ] == exclude


def test_version_is_above_the_newest_release_tag():
    # A shallow or tagless clone builds a version below the newest release,
    # 0.1.0.dev1+g..., which still parses; this is what catches it
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    match, exclude = describe_filters()
    found = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"]
        + [f"--match={p}" for p in match]
        + [f"--exclude={p}" for p in exclude],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if found.returncode != 0:
        if os.environ.get("CI"):
            pytest.fail(
                "no release tag is reachable: the checkout needs fetch-depth: 0"
            )
        pytest.skip("no release tag is reachable from this checkout")
    tag = found.stdout.strip()
    assert Version(HARK.__version__) >= Version(tag), (
        f"HARK's version, {HARK.__version__}, is below {tag}, "
        "the newest release tag reachable from this checkout"
    )
