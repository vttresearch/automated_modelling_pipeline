"""
Static scan of the repo's working tree for hardcoded secrets and leaked
internal VTT infrastructure references (e.g. the internal MLflow/DB server
IP, or the private GitLab URL turning up somewhere it shouldn't).

This is a *working-tree* scan only (it inspects files as they currently
are, not git history) - it does not catch secrets that were committed in
the past and later removed. If a real secret is ever found here or in
history, rotate/revoke it; do not rely on this test as the only defense.

Stdlib-only by design (os / re / pathlib) so it runs anywhere pytest runs,
with no extra dependencies.
"""
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories that are never scanned: build/cache artifacts, the OSS export
# staging output (itself derived from the very files this test scans), and
# test data fixtures (CSV/pickle, not source).
SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "build",
    "dist",
    "data",  # tests/data, examples/*/data - fixtures, not source
}
SKIP_DIR_SUFFIXES = (".egg-info",)
SKIP_RELATIVE_DIRS = {
    Path("tools/oss_export/stage"),
}

# This test module itself intentionally contains the strings/patterns it
# searches for (in comments, docstrings, and regex literals) - exclude it
# from its own scan to avoid self-inflicted false positives.
SKIP_RELATIVE_FILES = {
    Path("tests/test_no_secrets.py"),
}

# The internal on-prem server. This one is unconditionally disallowed,
# everywhere, no exceptions - unlike gitlab.vtt.fi (see below), there is no
# legitimate reason for this IP to appear in the repo.
INTERNAL_SERVER_IP = "193.166.160.215"

# Files where a plain `gitlab.vtt.fi` reference is expected/legitimate
# because they are the private repo's own self-references (its real git
# remote, packaging metadata, etc.) - NOT a leak. Anything else containing
# `gitlab.vtt.fi` is flagged. Paths are repo-root-relative, POSIX-style.
GITLAB_URL_ALLOWLIST = {
    "setup.py",
    "README.md",
    "web_api/requirements.txt",
    "web_api/Dockerfile_fastapi",
    "amp/kaukovainio/v3/modelling/kalman_mpc/rc_2R2C_daynight_mpc/train_mpc.py",
    "amp/kaukovainio/v3/modelling/kalman_mpc/rc_2R2C_daynight_mpc/train_heatm_sarimax.py",
    "amp/kaukovainio/v3/modelling/kalman_mpc/rc_2R2C_daynight_mpc/train_tempm.py",
    "amp/kaukovainio/v3/modelling/kalman/rc_2R2C_daynight/train_model.py",
    "amp/kaukovainio/v3/modelling/kalman/rc_1R1C_daynight/train_model.py",
    "amp/statsmodels/modelling/sarimax/train_model.py",
    # Documentation example text (private-repo-only clone instructions);
    # the OSS export rewrites this to the public URL - see
    # tools/oss_export/rewrite_public_refs.sh.
    "examples/notebooks/Electricity forecasting.ipynb",
}

GITLAB_URL_RE = re.compile(r"gitlab\.vtt\.fi")

# A credential embedded directly in a URL, e.g. https://user:token@host or
# https://token@host. This is the pattern that caused the real leak found
# in web_api/requirements.txt during the OSS security review.
EMBEDDED_URL_CREDENTIAL_RE = re.compile(
    r"https?://[^/\s@'\"]+:[^/\s@'\"]+@|https?://[A-Za-z0-9_\-]{16,}@"
)

# Hardcoded-secret patterns: password/token/api-key assignments, private
# key headers, and common cloud-credential shapes.
SECRET_PATTERNS = [
    re.compile(r"(?i)\bpassword\s*[:=]\s*['\"][^'\"\s]{4,}['\"]"),
    re.compile(r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?key|auth[_-]?token)\s*[:=]\s*['\"][^'\"\s]{4,}['\"]"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    EMBEDDED_URL_CREDENTIAL_RE,
]


def _is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
        return b"\x00" in chunk
    except OSError:
        return True


def _iter_source_files():
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT)
        if any(part in SKIP_DIR_NAMES for part in rel.parts[:-1]):
            continue
        if any(part.endswith(SKIP_DIR_SUFFIXES) for part in rel.parts[:-1]):
            continue
        if any(
            rel == skip or skip in rel.parents
            for skip in SKIP_RELATIVE_DIRS
        ):
            continue
        if rel in SKIP_RELATIVE_FILES:
            continue
        if _is_binary(path):
            continue
        yield rel, path


def _read_lines(path: Path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.readlines()
    except OSError:
        return []


def test_no_internal_server_ip():
    """The internal on-prem server IP must never appear anywhere, in any
    file, with no exceptions (unlike gitlab.vtt.fi, there is no legitimate
    reason for it to be hardcoded)."""
    offenders = []
    for rel, path in _iter_source_files():
        for lineno, line in enumerate(_read_lines(path), start=1):
            if INTERNAL_SERVER_IP in line:
                offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Found hardcoded internal server IP "
        f"({INTERNAL_SERVER_IP}) in:\n" + "\n".join(offenders)
    )


def test_no_hardcoded_secrets():
    """No hardcoded passwords/tokens/keys, or credentials embedded directly
    in a URL, anywhere in the repo. Applies everywhere - there is never a
    legitimate reason to commit a real secret, so no allowlist here."""
    offenders = []
    for rel, path in _iter_source_files():
        for lineno, line in enumerate(_read_lines(path), start=1):
            for pattern in SECRET_PATTERNS:
                if pattern.search(line):
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")
                    break

    assert not offenders, (
        "Found what looks like a hardcoded secret or embedded URL "
        "credential in:\n" + "\n".join(offenders) +
        "\n\nIf this is a false positive, tighten SECRET_PATTERNS in "
        "tests/test_no_secrets.py. If it's real, rotate/revoke the "
        "credential immediately - it is likely already in git history."
    )


def test_no_leaked_internal_gitlab_url():
    """gitlab.vtt.fi may legitimately appear in a small set of files that
    are the private repo's own self-references (git remote, packaging
    metadata). Anywhere else it is a leak. Even in an allowlisted file, an
    embedded credential next to the URL is still flagged."""
    offenders = []
    for rel, path in _iter_source_files():
        rel_posix = rel.as_posix()
        for lineno, line in enumerate(_read_lines(path), start=1):
            if not GITLAB_URL_RE.search(line):
                continue
            if rel_posix not in GITLAB_URL_ALLOWLIST:
                offenders.append(f"{rel_posix}:{lineno}: {line.strip()}")
            elif EMBEDDED_URL_CREDENTIAL_RE.search(line):
                offenders.append(
                    f"{rel_posix}:{lineno}: (allowlisted file, but embedded "
                    f"credential found) {line.strip()}"
                )

    assert not offenders, (
        "Found gitlab.vtt.fi reference outside the expected self-reference "
        "files (or with an embedded credential). If this is a new, "
        "legitimate self-reference, add the file path to "
        "GITLAB_URL_ALLOWLIST in tests/test_no_secrets.py. If it's a leak "
        "or a credential, fix/rotate it:\n" + "\n".join(offenders)
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
