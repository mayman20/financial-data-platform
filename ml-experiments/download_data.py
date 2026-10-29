#!/usr/bin/env python3
"""Download the pinned public datasets used by ml-experiments.

Opt-in helper. Importing this module does nothing; running

    python3 download_data.py --dataset all

fetches only the files pinned in DATA_SOURCES.json, checks byte hashes before
anything is parsed or written, and places verified files in the folder whose
code reads them (sentiment/ or news_price/). Each folder also gets a
DATA_PROVENANCE.json that ties the local files back to DATA_SOURCES.json.

Existing files are reused only when they match the pin. A mismatching file is
reported and left untouched; this script never overwrites, never extracts a
whole archive, and never runs downloaded content.

Standard library only. Written with AI assistance as part of the
October 2026 educational experiments.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
SOURCES_NAME = "DATA_SOURCES.json"
PROVENANCE_NAME = "DATA_PROVENANCE.json"

# DATA_SOURCES.json key -> folder (relative to this file) whose code reads it.
DATASET_DIRS = {"sentiment": "sentiment", "news_price": "news_price"}
CHOICES = {
    "sentiment": ("sentiment",),
    "news-price": ("news_price",),
    "all": ("sentiment", "news_price"),
}

ALLOWED_HOSTS = {"huggingface.co", "raw.githubusercontent.com"}
MAX_ARCHIVE_BYTES = 5 * 1024 * 1024
MAX_MEMBER_BYTES = 4 * 1024 * 1024
TIMEOUT_SECONDS = 60
USER_AGENT = "ml-experiments-download-data/1"


class DownloadError(Exception):
    """A pin, network, or local-file check failed. Nothing was overwritten."""


@dataclass(frozen=True)
class Target:
    filename: str
    sha256: str
    size: Optional[int]
    url: str
    archive_sha256: Optional[str] = None
    member: Optional[str] = None


# --- pin validation ---------------------------------------------------------

def _field(entry, name, label):
    if not isinstance(entry, dict) or name not in entry:
        raise DownloadError(f"{SOURCES_NAME}: missing {label}.{name}")
    return entry[name]


def _check_sha(value, label):
    if not (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value)):
        raise DownloadError(f"{SOURCES_NAME}: {label} is not a lowercase sha256 digest")
    return value


def _check_name(name):
    if (not isinstance(name, str) or not name or name.startswith(".")
            or "/" in name or "\\" in name or name == PROVENANCE_NAME):
        raise DownloadError(f"{SOURCES_NAME}: unsafe target filename {name!r}")
    return name


def _check_url(url):
    if not isinstance(url, str):
        raise DownloadError(f"{SOURCES_NAME}: URL must be a string")
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS:
        raise DownloadError(f"refusing URL outside the pinned hosts: {url}")
    return url


def plan_sentiment(entry):
    member = _field(entry, "member", "sentiment")
    if not isinstance(member, str) or not member:
        raise DownloadError(f"{SOURCES_NAME}: sentiment.member must be a string")
    return [Target(
        filename=_check_name(_field(entry, "filename", "sentiment")),
        sha256=_check_sha(_field(entry, "raw_sha256", "sentiment"), "sentiment.raw_sha256"),
        size=None,
        url=_check_url(_field(entry, "url", "sentiment")),
        archive_sha256=_check_sha(_field(entry, "archive_sha256", "sentiment"),
                                  "sentiment.archive_sha256"),
        member=member,
    )]


def plan_news_price(entry):
    files = _field(entry, "files", "news_price")
    if not isinstance(files, dict) or not files:
        raise DownloadError(f"{SOURCES_NAME}: news_price.files must be a non-empty object")
    targets = []
    for name in sorted(files):
        spec = files[name]
        size = _field(spec, "bytes", f"news_price.files.{name}")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise DownloadError(f"{SOURCES_NAME}: news_price.files.{name}.bytes must be positive")
        targets.append(Target(
            filename=_check_name(name),
            sha256=_check_sha(_field(spec, "sha256", f"news_price.files.{name}"),
                              f"news_price.files.{name}.sha256"),
            size=size,
            url=_check_url(_field(spec, "url", f"news_price.files.{name}")),
        ))
    return targets


PLANNERS = {"sentiment": plan_sentiment, "news_price": plan_news_price}


def load_sources(root):
    path = root / SOURCES_NAME
    try:
        raw = path.read_bytes()
        sources = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise DownloadError(f"cannot read {path}: {exc}") from exc
    if not isinstance(sources, dict):
        raise DownloadError(f"{SOURCES_NAME}: top level must be an object")
    return sources, hashlib.sha256(raw).hexdigest()


# --- hashing and local files ------------------------------------------------

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bytes(data, sha256, size, label):
    if size is not None and len(data) != size:
        raise DownloadError(f"{label}: expected {size} bytes, got {len(data)}")
    actual = hashlib.sha256(data).hexdigest()
    if actual != sha256:
        raise DownloadError(f"{label}: sha256 mismatch (expected {sha256}, got {actual})")


def already_pinned(path, target):
    """True if path already holds the pinned bytes, False if absent.

    Raises on anything else so a conflicting file is never replaced.
    """
    if path.is_symlink():
        raise DownloadError(f"{path} is a symlink; remove it or move it aside")
    if not path.exists():
        return False
    if not path.is_file():
        raise DownloadError(f"{path} exists and is not a regular file")
    if target.size is not None and path.stat().st_size != target.size:
        raise DownloadError(f"{path} exists with a different size; left untouched")
    if sha256_file(path) != target.sha256:
        raise DownloadError(f"{path} exists with a different sha256; left untouched")
    return True


def write_new(path, data):
    """Create path containing data. Never replaces an existing file."""
    try:
        fh = open(path, "xb")
    except FileExistsError as exc:
        raise DownloadError(f"{path} appeared during the run; left untouched") from exc
    try:
        with fh:
            fh.write(data)
            fh.flush()
    except BaseException:
        # Only removes the file this call just created.
        with contextlib.suppress(OSError):
            path.unlink()
        raise


# --- provenance -------------------------------------------------------------

def provenance_record(key, entry, sources_sha256, targets):
    files = {}
    for t in targets:
        record = {"sha256": t.sha256, "url": t.url}
        if t.size is not None:
            record["bytes"] = t.size
        if t.member is not None:
            record["archive_sha256"] = t.archive_sha256
            record["archive_member"] = t.member
        files[t.filename] = record
    return {
        "dataset": key,
        "generated_by": "download_data.py",
        "data_sources": {
            "file": f"../{SOURCES_NAME}",
            "sha256": sources_sha256,
            "entry": entry,
        },
        "files": files,
        "note": "Files above were hash-verified against the pins before being "
                "written. Third-party data; see the license fields in the entry.",
    }


def provenance_bytes(record):
    return (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")


def provenance_matches(path, record):
    """True if an equivalent provenance file exists, False if absent; raise on conflict."""
    if path.is_symlink():
        raise DownloadError(f"{path} is a symlink; remove it or move it aside")
    if not path.exists():
        return False
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise DownloadError(f"{path} exists but is not readable JSON; left untouched") from exc
    if existing != record:
        raise DownloadError(
            f"{path} records different sources; left untouched. "
            "Move it aside if DATA_SOURCES.json was intentionally changed."
        )
    return True


# --- network ----------------------------------------------------------------

def fetch(url, max_bytes):
    """Return at most max_bytes from a pinned HTTPS URL, or raise."""
    request = urllib.request.Request(_check_url(url), headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            final_url = response.geturl()
            if isinstance(final_url, str) and urllib.parse.urlsplit(final_url).scheme != "https":
                raise DownloadError(f"{url} redirected to a non-HTTPS URL")
            data = response.read(max_bytes + 1)
    except DownloadError:
        raise
    except (OSError, ValueError) as exc:
        raise DownloadError(f"download failed for {url}: {exc}") from exc
    if len(data) > max_bytes:
        raise DownloadError(f"{url} returned more than {max_bytes} bytes; refusing")
    return data


def read_pinned_member(archive, member):
    """Read one named member from verified archive bytes. Never extracts."""
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            try:
                info = zf.getinfo(member)
            except KeyError as exc:
                raise DownloadError(f"archive has no member {member!r}") from exc
            if info.flag_bits & 0x1:
                raise DownloadError(f"archive member {member!r} is encrypted")
            if info.file_size > MAX_MEMBER_BYTES:
                raise DownloadError(f"archive member {member!r} is larger than expected")
            with zf.open(info) as fh:
                data = fh.read(MAX_MEMBER_BYTES + 1)
    except zipfile.BadZipFile as exc:
        raise DownloadError(f"archive could not be read: {exc}") from exc
    if len(data) > MAX_MEMBER_BYTES:
        raise DownloadError(f"archive member {member!r} is larger than expected")
    return data


def download_target(target):
    """Fetch and verify one target, returning bytes that match the pin."""
    if target.member is None:
        data = fetch(target.url, target.size)
        verify_bytes(data, target.sha256, target.size, target.filename)
        return data
    archive = fetch(target.url, MAX_ARCHIVE_BYTES)
    # Hash the archive before handing it to zipfile.
    verify_bytes(archive, target.archive_sha256, None, f"{target.filename} archive")
    data = read_pinned_member(archive, target.member)
    verify_bytes(data, target.sha256, None, target.filename)
    return data


# --- driver -----------------------------------------------------------------

def prepare(root, keys, sources, sources_sha256):
    """Check every local target before any network access."""
    jobs = []
    for key in keys:
        entry = sources.get(key)
        if not isinstance(entry, dict):
            raise DownloadError(f"{SOURCES_NAME}: missing {key!r} entry")
        targets = PLANNERS[key](entry)
        directory = root / DATASET_DIRS[key]
        missing = [t for t in targets if not already_pinned(directory / t.filename, t)]
        record = provenance_record(key, entry, sources_sha256, targets)
        has_provenance = provenance_matches(directory / PROVENANCE_NAME, record)
        jobs.append((key, entry, directory, targets, missing, record, has_provenance))
    return jobs


def run(root, dataset, out=sys.stdout):
    if dataset not in CHOICES:
        raise DownloadError(f"unknown dataset {dataset!r}")
    sources, sources_sha256 = load_sources(root)
    jobs = prepare(root, CHOICES[dataset], sources, sources_sha256)

    for key, entry, directory, targets, missing, record, has_provenance in jobs:
        print(f"[{key}] source: {entry.get('source', 'see DATA_SOURCES.json')}", file=out)
        notice = entry.get("license") or entry.get("license_note")
        if notice:
            print(f"[{key}] license: {notice}", file=out)
        for target in targets:
            path = directory / target.filename
            if target not in missing:
                print(f"[{key}] reusing verified {path.relative_to(root)}", file=out)
                continue
            data = download_target(target)
            directory.mkdir(exist_ok=True)
            write_new(path, data)
            print(f"[{key}] saved verified {path.relative_to(root)}", file=out)
        if not has_provenance:
            write_new(directory / PROVENANCE_NAME, provenance_bytes(record))
        print(f"[{key}] provenance: {(directory / PROVENANCE_NAME).relative_to(root)}", file=out)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download the pinned public datasets for ml-experiments "
                    "(hash-verified, never overwrites).")
    parser.add_argument("--dataset", choices=sorted(CHOICES), default="all",
                        help="which dataset to fetch (default: all)")
    args = parser.parse_args(argv)
    try:
        run(HERE, args.dataset)
    except DownloadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
