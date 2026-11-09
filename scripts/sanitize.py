"""Deterministic, value-free scan of deliverable files and every new commit diff.

Usage: python scripts/sanitize.py --base main
Local environments, build output, Git internals and tool caches are not deliverables.
Ignored dotenv files outside those directories are still rejected. Findings print
only file/commit, line and category, never matched values. No credentials are read.
This lexical gate cannot establish absence of every possible unknown secret.
"""

from __future__ import annotations

import argparse
import ipaddress
import re
import subprocess
from pathlib import Path

SKIP = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    ".mplconfig",
    "site",
    "__pycache__",
    "build",
    "dist",
}
PATTERNS = {
    "personal path": re.compile(r"/(?:Us[e]rs|ho[m]e)/"),
    "email": re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "wallet": re.compile(r"0[xX][0-9a-fA-F]{40}\b"),
    "private key": re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    "provider key": re.compile(
        r"(?:sk[-_](?:proj[-_])?[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"
    ),
    "telegram credential": re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{30,}\b"),
    "assigned secret": re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|secret|password|authorization)[\"']?\s*[:=]\s*[\"'](?:Bearer\s+)?[A-Za-z0-9_+/=-]{16,}[\"']"
    ),
    "account identifier": re.compile(
        r"(?i)(?:account[_ -]?(?:id|number|no)|chat[_ -]?id|telegram[_ -]?id)[\"']?\s*[:=]\s*[\"']?[-+]?\d{6,}"
    ),
    "private hostname": re.compile(
        r"(?i)\b(?:clai"
        + r"re|dub"
        + r"lin)(?:[-.][A-Za-z0-9.-]+)?\b|\b[A-Za-z0-9.-]+\.(?:internal|localdomain)\b"
    ),
}
IP_CANDIDATE = re.compile(
    r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])|(?<![\w:])(?:[0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}(?![\w:.-])"
)


def git(*args: str) -> bytes:
    # Git metadata/diffs only; no process, auth, environment or credential inspection.
    return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)


def scan_text(label: str, text: str) -> list[str]:
    findings = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for category, pattern in PATTERNS.items():
            if pattern.search(line):
                findings.append(f"{label}:{lineno}: {category}")
        for match in IP_CANDIDATE.finditer(line):
            try:
                ipaddress.ip_address(match.group())
            except ValueError:
                continue
            findings.append(f"{label}:{lineno}: IP address")
    return findings


def run(base: str) -> int:
    root = Path(git("rev-parse", "--show-toplevel").decode().strip())
    tracked = {p for p in git("ls-files", "-z").decode().split("\0") if p}
    untracked = {
        p for p in git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0") if p
    }
    candidates = tracked | untracked

    # Find ignored dotenv files too, without entering local dependency/build folders.
    def dotenv_files(folder):
        for path in sorted(folder.iterdir()):
            if path.name in SKIP or path.name.endswith(".egg-info") or path.is_symlink():
                continue
            if path.is_dir():
                yield from dotenv_files(path)
            elif path.name == ".env" or path.name.startswith(".env.") or path.name.endswith(".env"):
                yield path.relative_to(root).as_posix()

    candidates.update(dotenv_files(root))
    findings, count = [], 0
    for relative in sorted(candidates):
        path = root / relative
        if path.is_symlink():
            findings.append(f"{relative}: symlink requires review")
            continue
        if not path.is_file():
            continue
        count += 1
        if path.name == ".env" or path.name.startswith(".env.") or path.name.endswith(".env"):
            findings.append(f"{relative}: prohibited dotenv file")
        if path.stat().st_size > 1_000_000:
            findings.append(f"{relative}: file exceeds 1 MB")
        try:
            data = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(f"{relative}: binary requires review")
            continue
        findings.extend(scan_text(relative, data))
    commits = git("rev-list", "--reverse", f"{base}..HEAD").decode().splitlines()
    for commit in commits:
        changed = (
            git("diff-tree", "--root", "--no-commit-id", "--name-only", "-r", commit)
            .decode()
            .splitlines()
        )
        for name in changed:
            part = Path(name).name
            if part == ".env" or part.startswith(".env.") or part.endswith(".env"):
                findings.append(f"commit {commit[:12]}: prohibited dotenv path")
            # Inspect the committed blob size even if removed in a later commit.
            exists = (
                subprocess.run(
                    ["git", "cat-file", "-e", f"{commit}:{name}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).returncode
                == 0
            )
            if exists and int(git("cat-file", "-s", f"{commit}:{name}")) > 1_000_000:
                findings.append(f"commit {commit[:12]}: committed file exceeds 1 MB")
        diff = git("diff-tree", "--root", "-p", "--no-ext-diff", "--no-commit-id", commit).decode()
        findings.extend(scan_text(f"commit {commit[:12]}", diff))
    for finding in findings:
        print(finding)
    print(
        f"Scanned {count} deliverable files and {len(commits)} new commit diffs: {len(findings)} findings"
    )
    return int(bool(findings))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="main")
    raise SystemExit(run(parser.parse_args().base))
