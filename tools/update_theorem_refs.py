#!/usr/bin/env python3
"""update_theorem_refs.py — scan, verify, and re-pin THEOREM-REF tags.

Home-agnostic, stdlib-only. Lives next to the tagged code (HARK branch); the
theorem repo (HAFiscal-Latest, private) is addressed via --repo / $THEOREM_REF_REPO.

HOME: ``tools/`` at the HARK repo root — HARK already keeps repo utility
scripts there (``tools/nb_exec.py``), while ``tests/`` holds unittest suites
only; a scan/re-pin CLI belongs with the former. Run from the repo root:

    python3 tools/update_theorem_refs.py --check
    python3 tools/update_theorem_refs.py --check --repo <theorem-repo-checkout>

The no-paths default scan (git-tracked *.py under the cwd, see below) covers
every tagged file in the repo, currently: HARK/ConsumptionSaving/pf_decay.py,
HARK/interpolation.py, HARK/ConsumptionSaving/ConsAggShockModel.py,
tests/ConsumptionSaving/test_pf_decay.py, tests/test_interpolation.py.
CI wrapper: tests/test_theorem_refs.py (degrades gracefully when the private
theorem repo is absent, e.g. on public econ-ark CI).

TAG GRAMMAR (one tag per line, inside a comment or docstring)
==============================================================

    # THEOREM-REF[<repo-name> @ <pin-sha> :: <file> :: <section> [:: <label>]]
    #   <1-3 line paraphrase of the referenced content, as indented
    #   comment continuation lines — not parsed by this tool>

Formally, the bracketed body is 3 or 4 fields separated by '::' :

    body    :=  repo WS? '@' WS? sha  '::'  file  '::'  section  [ '::' label ]
    repo    :=  human-readable repo name; no '@' or '::'  (e.g. HAFiscal-Latest)
    sha     :=  7-40 hex chars — the theorem-repo commit the paraphrase was
                written against ("the pin"). NEVER a line number.
    file    :=  path relative to the theorem-repo root
                (e.g. theory/powerlaw-decay/final_proof.md)
    section :=  heading text, verbatim enough to grep; may carry a section
                number and quotes, e.g.:  §0 "What is q*?"   or   3.4 Theorem I
    label   :=  OPTIONAL equation/theorem label expected INSIDE that section,
                e.g.:  eq (E)   |   Theorem γ-A   |   Prop A0   |   §6.1

A tag is MALFORMED (reported with file:line) when the token appears but the
body cannot be parsed: no [...] on the line, wrong field count, missing '@',
non-hex or wrong-length sha, or an empty field.

RESOLUTION (what "the tag resolves at SHA X" means)
===================================================
Content is read with `git -C <repo> show <X>:<file>` — the WORKTREE IS NEVER
READ OR WRITTEN. A tag resolves at X iff:
  (i)   <file> exists at X;
  (ii)  <section> fuzzy-matches an ATX heading (or a MyST `(anchor)=` line)
        outside code fences. Matching is done on normalized text (unicode
        NFKC, markdown backslash-escapes removed, curly quotes -> ascii,
        backticks stripped, whitespace collapsed, casefolded) using these
        needles, any of which may hit with non-word boundaries on both ends:
          - each double-quoted segment of the section field ("What is q*?"),
          - the whole field, the field minus quote marks, the field minus '§',
          - the field with a leading section number (§0 / 3.4 / §6.1.) stripped.
        Boundary-aware matching means 'Theorem I' does NOT match 'Theorem II'.
  (iii) if a <label> is present, it appears (same boundary-aware normalized
        search) within the matched section: from its heading line to the next
        heading of the same or higher level. Label needles: the whole label
        plus any parenthesized group in it (so `eq (E)` also matches `(E)`).
When several headings match (ii), the first (in document order) whose section
satisfies (iii) is used.

MATERIAL-CHANGE SIGNAL
======================
For each bumped tag, the sha256 of the section body normalized as ONE blob
(all whitespace incl. newlines collapsed) is compared between the old pin and
the new sha. Reflowed lines / cosmetic whitespace do NOT trigger it; wording
changes do. It is a cheap signal, not a semantic diff.

MODES, POLICY, EXIT CODES
=========================
update mode:   update_theorem_refs.py [paths...] --repo P --new-sha S [--write]
  POLICY: a tag is bumped to S iff file+section(+label) ALL resolve at S.
    - resolves at S, section body hash unchanged vs old pin  -> BUMP
    - resolves at S, body hash changed OR old pin unreadable -> REVIEW
      (STILL BUMPED — flagged for a human paraphrase-diff against the source)
    - does not resolve at S -> UNRESOLVED: left untouched, human must re-target
    - already pinned at S (same literal) -> OK (idempotent no-op)
  Without --write it is a dry run (prints what would change). REVIEW does not
  affect the exit code; UNRESOLVED and MALFORMED do.

check mode:    update_theorem_refs.py [paths...] --check [--repo P] [--strict]
  Tag SYNTAX is verified for all tags always. If the theorem repo is
  available (--repo or $THEOREM_REF_REPO points at a git repo), each tag is
  additionally verified to resolve at its CURRENT pin. If the repo is
  UNAVAILABLE (the normal case on public econ-ark/HARK CI, which does not
  have the private theorem repo): degrade gracefully — syntax-check only,
  print a NOTICE, exit 0. Pass --strict (private-side CI) to make an
  unavailable repo exit 1 instead.

Exit codes: 0 clean (REVIEW allowed) | 1 findings (malformed / unresolvable /
missing pin / unavailable-repo under --strict) | 2 usage error.

Default scan set (no paths given): git-tracked *.py under the cwd
(`git ls-files -- *.py`); falls back to a recursive *.py glob with a notice
if the cwd is not a git repo. Explicit file arguments are scanned regardless
of extension; directory arguments are scanned recursively for *.py. This
tool's own file (and its selftest) are skipped by basename so the grammar
examples above are not scanned as live tags.

Known limitations: the <repo-name> field is a human-facing name and is NOT
validated against --repo (worktrees/clones have arbitrary basenames); tags
must fit on one line; setext (underlined) markdown headings are not scanned.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import unicodedata

TOOL_BASENAMES_SKIP = ("update_theorem_refs.py", "selftest_update_theorem_refs.py")

TAG_TOKEN = "THEOREM-REF"
_TAG_RE = re.compile(re.escape(TAG_TOKEN) + r"\[(?P<body>.*)\]")
_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})\s+(\S.*)$")
_ANCHOR_RE = re.compile(r"^\(([^)\s]+)\)=\s*$")  # MyST target line
_FENCE_RE = re.compile(r"^ {0,3}(```|~~~)")

# ----------------------------------------------------------------------------
# text normalization + fuzzy matching
# ----------------------------------------------------------------------------

_QUOTE_MAP = {ord(c): '"' for c in "“”„‟«»"}
_QUOTE_MAP.update({ord(c): "'" for c in "‘’‚‛"})
_MD_UNESCAPE_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!<>|~\"'§])")


def normalize(s: str) -> str:
    """NFKC, markdown-unescape, ascii quotes, no backticks, one-space ws, casefold."""
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_QUOTE_MAP)
    s = _MD_UNESCAPE_RE.sub(r"\1", s)
    s = s.replace("`", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s.casefold()


def _boundary_search(needle: str, haystack: str) -> bool:
    """Substring search demanding non-word boundaries (so 'theorem i' != 'theorem ii')."""
    if not needle:
        return False
    pat = r"(?<!\w)" + re.escape(needle) + r"(?!\w)"
    return re.search(pat, haystack, flags=re.UNICODE) is not None


def section_needles(section_field: str) -> list:
    """Candidate normalized needles for the section field (see module docstring)."""
    fn = normalize(section_field)
    needles = []
    needles.extend(re.findall(r'"([^"]+)"', fn))           # quoted segments
    needles.append(fn)                                       # whole field
    nq = re.sub(r"\s+", " ", fn.replace('"', " ")).strip()   # minus quote marks
    needles.append(nq)
    needles.append(re.sub(r"\s+", " ", nq.replace("§", " ")).strip())  # minus §
    stripped = re.sub(r"^§?\s*\d+(?:\.\d+)*\.?\s*", "", nq)  # minus leading sec number
    needles.append(stripped)
    out = []
    for n in needles:
        n = n.strip()
        if len(n) >= 2 and n not in out:
            out.append(n)
    return out


def label_needles(label_field: str) -> list:
    fn = normalize(label_field)
    needles = [fn]
    for grp in re.findall(r"\(([^)]+)\)", fn):
        g = "(" + grp.strip() + ")"
        if g not in needles:
            needles.append(g)
    return [n for n in needles if n]


# ----------------------------------------------------------------------------
# tag parsing
# ----------------------------------------------------------------------------

class Tag:
    __slots__ = ("path", "lineno", "line", "repo_name", "sha", "sha_pos",
                 "file", "section", "label")

    def __init__(self, path, lineno, line, repo_name, sha, sha_pos,
                 file, section, label):
        self.path = path
        self.lineno = lineno
        self.line = line
        self.repo_name = repo_name
        self.sha = sha
        self.sha_pos = sha_pos          # char offset of sha within the line
        self.file = file
        self.section = section
        self.label = label              # may be None

    def loc(self):
        return "%s:%d" % (self.path, self.lineno)


class Malformed:
    __slots__ = ("path", "lineno", "reason")

    def __init__(self, path, lineno, reason):
        self.path = path
        self.lineno = lineno
        self.reason = reason

    def loc(self):
        return "%s:%d" % (self.path, self.lineno)


def parse_line(path: str, lineno: int, line: str):
    """Return Tag, Malformed, or None (line does not mention the token)."""
    if TAG_TOKEN not in line:
        return None
    m = _TAG_RE.search(line)
    if not m:
        return Malformed(path, lineno, "token present but no %s[...] tag on the line"
                         % TAG_TOKEN)
    body = m.group("body")
    parts = [p.strip() for p in body.split("::")]
    if len(parts) < 3:
        return Malformed(path, lineno,
                         "expected 3-4 '::'-separated fields (repo @ sha :: file :: "
                         "section [:: label]), got %d" % len(parts))
    if len(parts) > 4:
        return Malformed(path, lineno,
                         "too many '::' fields (%d); max is 4" % len(parts))
    head = parts[0]
    if "@" not in head:
        return Malformed(path, lineno, "first field lacks '@' (need: repo @ pin-sha)")
    repo_name, _, sha = head.partition("@")
    repo_name, sha = repo_name.strip(), sha.strip()
    if not repo_name:
        return Malformed(path, lineno, "empty repo name before '@'")
    if not _SHA_RE.match(sha):
        return Malformed(path, lineno, "pin %r is not a 7-40 char hex sha" % sha)
    file_, section = parts[1], parts[2]
    label = parts[3] if len(parts) == 4 else None
    if not file_:
        return Malformed(path, lineno, "empty file field")
    if not section:
        return Malformed(path, lineno, "empty section field")
    if label is not None and not label:
        return Malformed(path, lineno, "empty label field (drop the trailing '::')")
    sha_pos = line.index(sha, m.start("body"))
    return Tag(path, lineno, line, repo_name, sha, sha_pos, file_, section, label)


def scan_file(path: str):
    tags, malformed = [], []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, start=1):
                r = parse_line(path, i, line.rstrip("\n"))
                if isinstance(r, Tag):
                    tags.append(r)
                elif isinstance(r, Malformed):
                    malformed.append(r)
    except OSError as e:
        malformed.append(Malformed(path, 0, "unreadable: %s" % e))
    return tags, malformed


def iter_target_files(paths):
    """Yield files to scan; raise UsageError on a bad path argument."""
    if not paths:
        try:
            out = subprocess.run(
                ["git", "ls-files", "-z", "--", "*.py"],
                capture_output=True, check=True).stdout
            files = [f.decode("utf-8", "replace") for f in out.split(b"\0") if f]
        except (subprocess.CalledProcessError, FileNotFoundError):
            sys.stderr.write("NOTICE: cwd is not a git repo; falling back to a "
                             "recursive *.py glob\n")
            files = []
            for root, _dirs, names in os.walk("."):
                files.extend(os.path.join(root, n) for n in names
                             if n.endswith(".py"))
            files.sort()
        for f in files:
            if os.path.basename(f) not in TOOL_BASENAMES_SKIP:
                yield f
        return
    for p in paths:
        if os.path.isfile(p):
            yield p
        elif os.path.isdir(p):
            hits = []
            for root, _dirs, names in os.walk(p):
                hits.extend(os.path.join(root, n) for n in sorted(names)
                            if n.endswith(".py")
                            and n not in TOOL_BASENAMES_SKIP)
            for h in sorted(hits):
                yield h
        else:
            raise UsageError("path argument does not exist: %s" % p)


# ----------------------------------------------------------------------------
# theorem-repo access (read-only: rev-parse / cat-file / show)
# ----------------------------------------------------------------------------

class UsageError(Exception):
    pass


class TheoremRepo:
    def __init__(self, path: str):
        self.path = path
        self._rev_cache = {}
        self._blob_cache = {}

    @staticmethod
    def open(path):
        """Return TheoremRepo if path is a usable git repo, else None."""
        if not path or not os.path.isdir(path):
            return None
        r = subprocess.run(["git", "-C", path, "rev-parse", "--git-dir"],
                           capture_output=True)
        return TheoremRepo(path) if r.returncode == 0 else None

    def _git(self, *args):
        return subprocess.run(["git", "-C", self.path] + list(args),
                              capture_output=True)

    def rev_parse(self, sha: str):
        """Full commit sha for `sha`, or None."""
        if sha in self._rev_cache:
            return self._rev_cache[sha]
        r = self._git("rev-parse", "--verify", "--quiet", sha + "^{commit}")
        full = r.stdout.decode().strip() if r.returncode == 0 else None
        self._rev_cache[sha] = full
        return full

    def read_file(self, sha: str, path: str):
        """File content at sha, or None if absent. sha must be resolvable."""
        key = (sha, path)
        if key in self._blob_cache:
            return self._blob_cache[key]
        r = self._git("cat-file", "-e", "%s:%s" % (sha, path))
        if r.returncode != 0:
            self._blob_cache[key] = None
            return None
        r = self._git("show", "%s:%s" % (sha, path))
        text = r.stdout.decode("utf-8", "replace") if r.returncode == 0 else None
        self._blob_cache[key] = text
        return text


# ----------------------------------------------------------------------------
# markdown section resolution
# ----------------------------------------------------------------------------

def parse_headings(text: str):
    """Return (lines, headings); heading = (line_idx, level, norm_text, anchors)."""
    lines = text.split("\n")
    headings = []
    in_fence = False
    pending_anchors = []
    for i, ln in enumerate(lines):
        if _FENCE_RE.match(ln):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        hm = _HEADING_RE.match(ln)
        if hm:
            headings.append((i, len(hm.group(1)), normalize(hm.group(2)),
                             tuple(pending_anchors)))
            pending_anchors = []
            continue
        am = _ANCHOR_RE.match(ln)
        if am:
            pending_anchors.append(normalize(am.group(1)))
    return lines, headings


def section_span(lines, headings, k):
    """(start, end) line indices of section k: heading to next same-or-higher."""
    start, level = headings[k][0], headings[k][1]
    end = len(lines)
    for j in range(k + 1, len(headings)):
        if headings[j][1] <= level:
            end = headings[j][0]
            break
    return start, end


class Resolution:
    __slots__ = ("ok", "reason", "heading", "body_hash")

    def __init__(self, ok, reason="", heading="", body_hash=""):
        self.ok = ok
        self.reason = reason
        self.heading = heading
        self.body_hash = body_hash


def resolve_tag_at(repo: TheoremRepo, tag: Tag, sha: str) -> Resolution:
    full = repo.rev_parse(sha)
    if full is None:
        return Resolution(False, "pin %s not found in theorem repo" % sha)
    text = repo.read_file(full, tag.file)
    if text is None:
        return Resolution(False, "file %r does not exist at %s" % (tag.file, sha))
    lines, headings = parse_headings(text)
    needles = section_needles(tag.section)
    matched = []
    for k, (_i, _lvl, htext, anchors) in enumerate(headings):
        for n in needles:
            if _boundary_search(n, htext) or any(
                    _boundary_search(n, a) for a in anchors):
                matched.append(k)
                break
    if not matched:
        return Resolution(False, "no heading matches section %r at %s"
                          % (tag.section, sha))
    lneedles = label_needles(tag.label) if tag.label else None
    for k in matched:
        s, e = section_span(lines, headings, k)
        body_norm = normalize("\n".join(lines[s:e]))
        if lneedles is not None and not any(
                _boundary_search(n, body_norm) for n in lneedles):
            continue
        return Resolution(True, heading=headings[k][2],
                          body_hash=hashlib.sha256(
                              body_norm.encode()).hexdigest()[:16])
    return Resolution(False,
                      "label %r not found inside section %r at %s "
                      "(%d heading(s) matched the section)"
                      % (tag.label, tag.section, sha, len(matched)))


# ----------------------------------------------------------------------------
# modes
# ----------------------------------------------------------------------------

def _report(status, loc, msg):
    print("%-11s %s  %s" % (status, loc, msg))


def run_check(tags, malformed, repo, strict):
    findings = len(malformed)
    for m in malformed:
        _report("MALFORMED", m.loc(), m.reason)
    if repo is None:
        for t in tags:
            _report("SYNTAX-OK", t.loc(),
                    "%s @ %s :: %s" % (t.repo_name, t.sha, t.file))
        print("NOTICE: theorem repo unavailable (no --repo/$THEOREM_REF_REPO or "
              "not a git repo); syntax-check only%s."
              % (" — FAILING due to --strict" if strict else ""))
        if strict:
            findings += 1
    else:
        for t in tags:
            res = resolve_tag_at(repo, t, t.sha)
            if res.ok:
                _report("OK", t.loc(), "resolves at %s (section %r)"
                        % (t.sha, res.heading))
            else:
                _report("UNRESOLVED", t.loc(), res.reason)
                findings += 1
    print("check: %d tag(s), %d malformed, %d finding(s) total"
          % (len(tags), len(malformed), findings))
    return 1 if findings else 0


def run_update(tags, malformed, repo, new_sha, write):
    findings = len(malformed)
    for m in malformed:
        _report("MALFORMED", m.loc(), m.reason)
    new_full = repo.rev_parse(new_sha)
    if new_full is None:
        raise UsageError("--new-sha %s not found in theorem repo %s"
                         % (new_sha, repo.path))
    edits = {}  # path -> list of (lineno, old_sha, sha_pos)
    n_bump = n_review = n_noop = 0
    for t in tags:
        res_new = resolve_tag_at(repo, t, new_sha)
        if not res_new.ok:
            _report("UNRESOLVED", t.loc(), res_new.reason + " — NOT bumped")
            findings += 1
            continue
        if t.sha == new_sha:
            _report("OK", t.loc(), "already pinned at %s" % new_sha)
            n_noop += 1
            continue
        res_old = resolve_tag_at(repo, t, t.sha)
        changed = (not res_old.ok) or (res_old.body_hash != res_new.body_hash)
        edits.setdefault(t.path, []).append((t.lineno, t.sha, t.sha_pos))
        verb = "bumped" if write else "would bump"
        if changed:
            why = (res_old.reason if not res_old.ok
                   else "section body CHANGED (normalized hash %s -> %s)"
                   % (res_old.body_hash, res_new.body_hash))
            _report("REVIEW", t.loc(), "%s %s -> %s; %s — re-verify the paraphrase"
                    % (verb, t.sha, new_sha, why))
            n_review += 1
        else:
            _report("BUMP", t.loc(), "%s %s -> %s (section body unchanged)"
                    % (verb, t.sha, new_sha))
            n_bump += 1
    if write:
        for path, todo in edits.items():
            with open(path, "r", encoding="utf-8") as fh:
                lines = fh.read().split("\n")
            for lineno, old_sha, sha_pos in todo:
                ln = lines[lineno - 1]
                assert ln[sha_pos:sha_pos + len(old_sha)] == old_sha, \
                    "internal error: sha moved in %s:%d" % (path, lineno)
                lines[lineno - 1] = (ln[:sha_pos] + new_sha
                                     + ln[sha_pos + len(old_sha):])
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines))
    print("update%s: %d tag(s): %d bumped, %d bumped-for-REVIEW, %d already "
          "pinned, %d unresolved, %d malformed"
          % ("" if write else " (dry-run)", len(tags), n_bump, n_review,
             n_noop, len(tags) - n_bump - n_review - n_noop, len(malformed)))
    return 1 if findings else 0


# ----------------------------------------------------------------------------
# cli
# ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="update_theorem_refs.py",
        description="Scan, verify, and re-pin THEOREM-REF tags "
                    "(see module docstring for the grammar and policy).")
    ap.add_argument("paths", nargs="*",
                    help="files or directories to scan (default: git-tracked "
                         "*.py under the cwd)")
    ap.add_argument("--repo", default=os.environ.get("THEOREM_REF_REPO"),
                    help="path to the theorem repo checkout "
                         "(default: $THEOREM_REF_REPO)")
    ap.add_argument("--new-sha", help="update mode: bump resolvable pins to "
                                      "this theorem-repo sha")
    ap.add_argument("--write", action="store_true",
                    help="update mode: apply the bumps (default: dry-run)")
    ap.add_argument("--check", action="store_true",
                    help="CI mode: verify syntax always, resolution at the "
                         "current pin when the repo is available")
    ap.add_argument("--strict", action="store_true",
                    help="--check: treat an unavailable theorem repo as a "
                         "failure (private-side CI)")
    args = ap.parse_args(argv)

    try:
        if args.check and args.new_sha:
            raise UsageError("--check and --new-sha are mutually exclusive")
        if not args.check and not args.new_sha:
            raise UsageError("pick a mode: --check, or --new-sha (update)")
        if args.write and not args.new_sha:
            raise UsageError("--write only applies to update mode (--new-sha)")
        if args.strict and not args.check:
            raise UsageError("--strict only applies to --check mode")

        tags, malformed = [], []
        for f in iter_target_files(args.paths):
            t, m = scan_file(f)
            tags.extend(t)
            malformed.extend(m)

        if args.check:
            repo = TheoremRepo.open(args.repo) if args.repo else None
            return run_check(tags, malformed, repo, args.strict)

        # update mode: the theorem repo is REQUIRED
        if not args.repo:
            raise UsageError("update mode needs --repo (or $THEOREM_REF_REPO)")
        repo = TheoremRepo.open(args.repo)
        if repo is None:
            raise UsageError("--repo %s is not a usable git repo" % args.repo)
        if not _SHA_RE.match(args.new_sha):
            raise UsageError("--new-sha %r is not a 7-40 char hex sha"
                             % args.new_sha)
        return run_update(tags, malformed, repo, args.new_sha, args.write)
    except UsageError as e:
        sys.stderr.write("usage error: %s\n" % e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
