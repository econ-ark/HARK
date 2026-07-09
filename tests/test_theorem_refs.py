"""CI wrapper for tools/update_theorem_refs.py (theorem-ref tag hygiene).

The power-law-decay PR carries theorem-ref tags — pinned citations into the
private HAFiscal-Latest theorem repo (theory/powerlaw-decay/) — in its comments
and docstrings. ``tools/update_theorem_refs.py --check`` verifies tag SYNTAX
always, and tag RESOLUTION (file + section heading + optional label exist at
the pinned commit) whenever a theorem-repo checkout is available.

Pre-registered expectations (declared before first run; never weakened):
  * WITHOUT the theorem repo (the public econ-ark CI case) the checker must
    DEGRADE GRACEFULLY: syntax-check only, print a NOTICE, exit 0.
  * It must find at least ``MIN_EXPECTED_TAGS`` tags (the PR landed 22; a
    collapse to fewer than 20 means tags were mass-deleted or the scanner
    broke) and 0 malformed.
  * WITH a local theorem-repo checkout (private-side dev boxes), every tag
    must additionally RESOLVE at its pin: exit 0 and no UNRESOLVED lines.

Note: this file deliberately never spells the uppercase tag token, so the
scanner has nothing to parse here.
"""

import os
import subprocess
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(REPO_ROOT, "tools", "update_theorem_refs.py")

# Known private-side checkout of the theorem repo (worktree pin 71ca7c61);
# $THEOREM_REF_REPO overrides. Absent on public CI — that path is the
# graceful-degradation test.
THEOREM_REPO = os.environ.get(
    "THEOREM_REF_REPO",
    "/home/shared/github/llorracc/HAFiscal-Latest/.worktrees/powerlaw-theorem",
)

MIN_EXPECTED_TAGS = 20


def _run_check(extra_args, env_overrides=None):
    env = dict(os.environ)
    env.pop("THEOREM_REF_REPO", None)  # the test controls repo availability
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, TOOL, "--check"] + extra_args,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


class TestTheoremRefsCheck(unittest.TestCase):
    def test_check_without_theorem_repo_degrades_gracefully(self):
        """No theorem repo: syntax-check only, NOTICE printed, exit 0."""
        r = _run_check([])
        self.assertEqual(
            r.returncode, 0,
            f"--check without the theorem repo must exit 0, got "
            f"{r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}",
        )
        self.assertIn("NOTICE", r.stdout, r.stdout)
        self.assertIn(" 0 malformed", r.stdout, r.stdout)
        n_tags = int(r.stdout.rsplit("check: ", 1)[1].split(" tag(s)")[0])
        self.assertGreaterEqual(
            n_tags, MIN_EXPECTED_TAGS,
            f"expected >= {MIN_EXPECTED_TAGS} tags, scanner found {n_tags}",
        )

    @unittest.skipUnless(
        os.path.isdir(os.path.join(THEOREM_REPO, ".git"))
        or os.path.isfile(os.path.join(THEOREM_REPO, ".git")),
        "theorem repo checkout not available (public CI)",
    )
    def test_check_with_theorem_repo_all_tags_resolve(self):
        """Theorem repo present: every tag resolves at its pin; exit 0."""
        r = _run_check(["--repo", THEOREM_REPO])
        self.assertEqual(
            r.returncode, 0,
            f"--check with the theorem repo must exit 0, got "
            f"{r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}",
        )
        self.assertNotIn("UNRESOLVED", r.stdout, r.stdout)
        self.assertNotIn("MALFORMED", r.stdout, r.stdout)
        self.assertIn(" 0 finding(s) total", r.stdout, r.stdout)


if __name__ == "__main__":
    unittest.main()
