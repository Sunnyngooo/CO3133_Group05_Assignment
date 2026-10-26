"""Checks for resume behavior, data preservation and leakage prevention."""

import io
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import requests

from data_utils import ROOT, metadata_groups, select_records
from download_images import download_one, inspect_image
from prepare_data import assign_splits, build_groups, validate_splits, write_manifests


def image_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (80, 60), (30, 90, 150)).save(stream, format="JPEG")
    return stream.getvalue()


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(dir=ROOT / "data/assignment3")
        self.path = Path(self.folder.name)
        self.record = {"image_id": "test_image", "section": "Health", "image_url": "https://static01.nyt.com/example.jpg"}

    def tearDown(self):
        self.path.resolve().relative_to((ROOT / "data/assignment3").resolve())
        self.folder.cleanup()

    def test_valid_existing_image_skips_network(self):
        (self.path / "test_image.jpg").write_bytes(image_bytes())
        with patch("download_images.session") as client:
            result = download_one(self.record, self.path, 1, 0)
            client.assert_not_called()
        self.assertEqual(result["status"], "cached")
        self.assertEqual((result["width"], result["height"]), (80, 60))

    def test_failed_replacement_preserves_existing_file(self):
        path = self.path / "test_image.jpg"
        path.write_bytes(b"existing-invalid-file")
        failure = requests.HTTPError("not found", response=Mock(status_code=404))
        with patch("download_images.session") as client:
            client.return_value.get.side_effect = failure
            result = download_one(self.record, self.path, 1, 2)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(path.read_bytes(), b"existing-invalid-file")

    def test_valid_download_is_verified_and_written(self):
        response = Mock(headers={})
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.iter_content.return_value = [image_bytes()]
        with patch("download_images.session") as client:
            client.return_value.get.return_value = response
            result = download_one(self.record, self.path, 1, 0)
        self.assertEqual(result["status"], "downloaded")
        self.assertEqual((self.path / "test_image.jpg").read_bytes(), image_bytes())
        self.assertFalse((self.path / "test_image.jpg.part").exists())


class GroupingTests(unittest.TestCase):
    def test_parallel_image_reading_preserves_duplicate_groups(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "data/assignment3") as directory:
            folder = Path(directory)
            folder.resolve().relative_to((ROOT / "data/assignment3").resolve())
            rows = []
            for index in range(3):
                path = folder / f"{index}.jpg"
                Image.new("RGB", (80, 60), (index // 2 * 100, 90, 150)).save(path)
                info = inspect_image(path.read_bytes())
                rows.append({
                    "image_id": str(index), "label": "A", "metadata_group": str(index),
                    "image_path": str(path.relative_to(ROOT)),
                    "pixel_sha256": info["pixel_sha256"], "file_sha256": info["file_sha256"],
                    "normalized_headline": str(index),
                    "image_url": f"https://img/{index}.jpg", "article_url": f"https://news/{index}",
                })
            sequential, sequential_pairs = build_groups([dict(row) for row in rows], workers=1)
            parallel, parallel_pairs = build_groups([dict(row) for row in rows], workers=4)
            self.assertEqual(sequential, parallel)
            self.assertEqual(sequential_pairs, parallel_pairs)
            self.assertEqual(sorted(map(len, parallel)), [1, 2])

    def test_full_selection_keeps_every_record_and_class_imbalance(self):
        rows = [
            {"image_id": str(index), "section": "A" if index < 7 else "B",
             "article_url": f"https://news/{index}", "image_url": f"https://img/{index}.jpg"}
            for index in range(10)
        ]
        selected = select_records(rows, 0, 44)
        self.assertEqual({row["image_id"] for row in selected}, {row["image_id"] for row in rows})
        self.assertEqual(sum(row["section"] == "A" for row in selected), 7)
        self.assertEqual(sum(row["section"] == "B" for row in selected), 3)

    def test_manifest_hash_is_hash_of_actual_file_bytes(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "data/assignment3") as directory:
            folder = Path(directory)
            folder.resolve().relative_to((ROOT / "data/assignment3").resolve())
            hashes = write_manifests(folder, {"train": [{"headline": "Tiêu đề", "image_id": "example"}]})
            saved = (folder / "train.jsonl").read_bytes()
            self.assertNotIn(b"\r\n", saved)
            self.assertEqual(hashes["train"], hashlib.sha256(saved).hexdigest())

    def test_split_checker_rejects_exact_text_overlap(self):
        splits = {}
        for index, name in enumerate(("train", "validation", "test")):
            unique = str(index)
            splits[name] = [{
                "image_id": unique, "label": "A", "group_id": unique,
                "pixel_sha256": unique, "file_sha256": unique,
                "canonical_image_url": unique, "canonical_article_url": unique,
                "normalized_headline": "repeated headline" if index < 2 else "different headline",
            }]
        with self.assertRaisesRegex(ValueError, "normalized_headline overlap"):
            validate_splits(splits, {"A"})

    def test_transitive_metadata_links_survive_query_variants(self):
        rows = [
            {"image_id": "a", "section": "A", "article_url": "https://news/a?q=1", "image_url": "https://img/one.jpg?a=1"},
            {"image_id": "b", "section": "A", "article_url": "https://news/b", "image_url": "https://img/one.jpg?a=2"},
            {"image_id": "c", "section": "A", "article_url": "https://news/b?q=2", "image_url": "https://img/other.jpg"},
        ]
        self.assertEqual(len(metadata_groups(rows)), 1)
        # A cap cannot split a 3-record duplicate component into partial records.
        self.assertEqual(select_records(rows, 2, 44), [])
        self.assertEqual(len(select_records(rows, 3, 44)), 3)

    def test_whole_group_split_has_no_overlap_and_all_classes(self):
        groups = []
        for index in range(40):
            group = []
            for label in ("A", "B"):
                sample_id = f"{index}_{label}"
                group.append({
                    "image_id": sample_id, "label": label, "group_id": f"group_{index}",
                    "pixel_sha256": sample_id, "file_sha256": sample_id,
                    "canonical_image_url": sample_id, "canonical_article_url": sample_id,
                    "normalized_headline": sample_id,
                })
            groups.append(group)
        splits = assign_splits(groups, 44)
        self.assertEqual(sum(map(len, splits.values())), 80)
        for rows in splits.values():
            self.assertEqual({row["label"] for row in rows}, {"A", "B"})
        validate_splits(splits, {"A", "B"})
        self.assertEqual(
            [[row["image_id"] for row in splits[name]] for name in splits],
            [[row["image_id"] for row in assign_splits(groups, 44)[name]] for name in splits],
        )


if __name__ == "__main__":
    unittest.main()
