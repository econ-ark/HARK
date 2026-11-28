"""
Build HARK's wheel at tags and one commit after them, in a clone of this
checkout, and check the Version in each wheel's METADATA against what
pyproject.toml's version scheme (setuptools_scm's guess-next-dev) gives:

- at a release tag, the tag itself;
- one commit after it, the next patch's first development version;
- one commit after a tag that starts a cycle, such as 99.2.0.dev0, that cycle's
  first development version;
- after a tag that tag_regex rejects, still counted from the last release tag.

The tags are made in the clone only, with versions no release will have. Run
from the repository's root, with build, setuptools and setuptools_scm
installed: python tools/check_wheel_versions.py
"""

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def git(*args, cwd):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def wheel_version(tree, out):
    """The Version in the METADATA of the wheel built from tree."""
    for old in Path(out).glob("*.whl"):
        old.unlink()
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", out],
        cwd=tree,
        check=True,
        capture_output=True,
    )
    (wheel,) = Path(out).glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        (metadata,) = [
            n for n in archive.namelist() if n.endswith(".dist-info/METADATA")
        ]
        for line in archive.read(metadata).decode().splitlines():
            if line.startswith("Version: "):
                return line[len("Version: ") :]
    raise RuntimeError(f"{wheel.name} has no Version")


def main():
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        clone, out = Path(tmp) / "HARK", Path(tmp) / "dist"
        subprocess.run(
            [
                "git",
                "-c",
                "advice.detachedHead=false",
                "clone",
                "--quiet",
                str(ROOT),
                str(clone),
            ],
            check=True,
        )
        identity = ["-c", "user.name=version check", "-c", "user.email=check@invalid"]

        def commit():
            git(
                *identity,
                "commit",
                "--allow-empty",
                "--quiet",
                "-m",
                "one commit",
                cwd=clone,
            )
            return git("rev-parse", "HEAD", cwd=clone)

        def check(label, public, sha=None):
            built = Version(wheel_version(clone, out))
            local_ok = (
                sha is None
                and built.local is None
                or (
                    sha is not None
                    and built.local is not None
                    and sha.startswith(built.local[1:])
                )
            )
            ok = built.public == public and local_ok
            print(f"{'ok  ' if ok else 'FAIL'} {label}: {built}")
            if not ok:
                failures.append(f"{label}: built {built}, expected {public}")

        git("tag", "99.1.0", cwd=clone)
        check("at the release tag 99.1.0", "99.1.0")
        check("one commit after it", "99.1.1.dev1", commit())
        git("tag", "99.2.0.dev0", cwd=clone)
        check("one commit after the cycle's tag 99.2.0.dev0", "99.2.0.dev1", commit())
        git("tag", "v99.9.9-not-a-release", cwd=clone)
        check("one more after a tag tag_regex rejects", "99.2.0.dev2", commit())
    if failures:
        sys.exit("\n".join(failures))


if __name__ == "__main__":
    main()
