"""Offline regression tests for the pinned-data preparation helper."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


MODULE_PATH = Path(__file__).with_name("download_data.py")
spec = importlib.util.spec_from_file_location("download_data", MODULE_PATH)
download_data = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = download_data
spec.loader.exec_module(download_data)


def digest(data):
    return hashlib.sha256(data).hexdigest()


class DownloadDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def sources(self, dataset, entry):
        raw = json.dumps({dataset: entry}, sort_keys=True).encode("utf-8")
        (self.root / download_data.SOURCES_NAME).write_bytes(raw)
        return digest(raw)

    def test_prepares_verified_fixture_and_writes_provenance(self):
        data = b"date,headline\n2026-01-01,fixture\n"
        entry = {"files": {"fixture.csv": {
            "url": "https://raw.githubusercontent.com/example/data.csv",
            "sha256": digest(data), "bytes": len(data),
        }}}
        source_hash = self.sources("news_price", entry)
        output = io.StringIO()
        with patch.object(download_data, "fetch", return_value=data) as fetch:
            download_data.run(self.root, "news-price", out=output)
        fetch.assert_called_once_with(entry["files"]["fixture.csv"]["url"], len(data))
        target = self.root / "news_price" / "fixture.csv"
        self.assertEqual(target.read_bytes(), data)
        record = json.loads((target.parent / "DATA_PROVENANCE.json").read_text())
        self.assertEqual(record["data_sources"]["sha256"], source_hash)
        self.assertEqual(record["files"]["fixture.csv"]["sha256"], digest(data))

    def test_second_run_reuses_verified_file_without_fetch(self):
        data = b"fixture bytes\n"
        entry = {"files": {"fixture.csv": {
            "url": "https://raw.githubusercontent.com/example/data.csv",
            "sha256": digest(data), "bytes": len(data),
        }}}
        self.sources("news_price", entry)
        with patch.object(download_data, "fetch", return_value=data):
            download_data.run(self.root, "news-price", out=io.StringIO())
        with patch.object(download_data, "fetch", side_effect=AssertionError("unexpected fetch")):
            download_data.run(self.root, "news-price", out=io.StringIO())
        self.assertEqual((self.root / "news_price" / "fixture.csv").read_bytes(), data)

    def test_conflicting_local_file_stops_before_any_fetch(self):
        expected = b"pinned\n"
        entry = {"files": {"fixture.csv": {
            "url": "https://raw.githubusercontent.com/example/data.csv",
            "sha256": digest(expected), "bytes": len(expected),
        }}}
        self.sources("news_price", entry)
        folder = self.root / "news_price"
        folder.mkdir()
        target = folder / "fixture.csv"
        target.write_bytes(b"local edits\n")
        with patch.object(download_data, "fetch") as fetch:
            with self.assertRaises(download_data.DownloadError):
                download_data.run(self.root, "news-price", out=io.StringIO())
        fetch.assert_not_called()
        self.assertEqual(target.read_bytes(), b"local edits\n")

    def test_bad_download_hash_writes_neither_data_nor_provenance(self):
        expected = b"expected"
        entry = {"files": {"fixture.csv": {
            "url": "https://raw.githubusercontent.com/example/data.csv",
            "sha256": digest(expected), "bytes": len(expected),
        }}}
        self.sources("news_price", entry)
        with patch.object(download_data, "fetch", return_value=b"tampered"):
            with self.assertRaisesRegex(download_data.DownloadError, "sha256 mismatch"):
                download_data.run(self.root, "news-price", out=io.StringIO())
        folder = self.root / "news_price"
        self.assertFalse((folder / "fixture.csv").exists())
        self.assertFalse((folder / "DATA_PROVENANCE.json").exists())

    def test_zip_reads_only_exact_pinned_member_without_extracting(self):
        selected = b"selected fixture\n"
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("nested/expected.txt", selected)
            archive.writestr("nested/other.txt", b"must not be selected")
        archive_bytes = stream.getvalue()
        entry = {
            "filename": "sentences.txt", "member": "nested/expected.txt",
            "url": "https://huggingface.co/example/data.zip",
            "archive_sha256": digest(archive_bytes), "raw_sha256": digest(selected),
        }
        self.sources("sentiment", entry)
        with patch.object(download_data, "fetch", return_value=archive_bytes) as fetch:
            download_data.run(self.root, "sentiment", out=io.StringIO())
        fetch.assert_called_once_with(entry["url"], download_data.MAX_ARCHIVE_BYTES)
        folder = self.root / "sentiment"
        self.assertEqual((folder / "sentences.txt").read_bytes(), selected)
        self.assertEqual(sorted(p.name for p in folder.iterdir()),
                         ["DATA_PROVENANCE.json", "sentences.txt"])

    def test_zip_rejects_archive_hash_mismatch_before_member_read(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("expected.txt", b"fixture")
        archive_bytes = stream.getvalue()
        entry = {
            "filename": "sentences.txt", "member": "expected.txt",
            "url": "https://huggingface.co/example/data.zip",
            "archive_sha256": "0" * 64, "raw_sha256": digest(b"fixture"),
        }
        self.sources("sentiment", entry)
        with patch.object(download_data, "fetch", return_value=archive_bytes):
            with self.assertRaisesRegex(download_data.DownloadError, "archive: sha256 mismatch"):
                download_data.run(self.root, "sentiment", out=io.StringIO())
        self.assertFalse((self.root / "sentiment" / "sentences.txt").exists())

    def test_zip_rejects_missing_exact_member(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("folder/expected.txt", b"fixture")
        archive_bytes = stream.getvalue()
        target = download_data.Target(
            "sentences.txt", digest(b"fixture"), None,
            "https://huggingface.co/example/data.zip", digest(archive_bytes), "expected.txt",
        )
        with patch.object(download_data, "fetch", return_value=archive_bytes):
            with self.assertRaisesRegex(download_data.DownloadError, "no member 'expected.txt'"):
                download_data.download_target(target)


if __name__ == "__main__":
    unittest.main()
