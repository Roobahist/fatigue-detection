"""Dependency-free regression tests for research dataset conventions."""

import tempfile
import unittest
from pathlib import Path

from Zh.src.utils.experiment_data import extract_target, load_kss_scores, resolve_data_paths


class DataPathTests(unittest.TestCase):
    def test_drozy_layout(self):
        paths = resolve_data_paths("DROZY", "/datasets")
        self.assertEqual(paths["kss_score_path"], "/datasets/DROZY/data/original/KSS.txt")
        self.assertTrue(paths["embeddings_dir"].startswith("/datasets/DROZY/data/videos/"))

    def test_uta_and_overrides(self):
        paths = resolve_data_paths("UTA", "/datasets", {"features_dir": "/custom/features"})
        self.assertEqual(paths["features_dir"], "/custom/features")
        self.assertIsNone(paths["kss_score_path"])

    def test_bad_config(self):
        for dataset, overrides in (("unknown", None), ("UTA", {"typo": "/tmp"}), ("UTA", {"features_dir": ""})):
            with self.subTest(dataset=dataset, overrides=overrides), self.assertRaises(ValueError):
                resolve_data_paths(dataset, "/datasets", overrides)


class TargetTests(unittest.TestCase):
    def test_uta(self):
        self.assertEqual(extract_target("/data/abc_def_10.0_clip.dat", "UTA"), 10.0)

    def test_drozy_one_based_indices(self):
        scores = ((1, 5, 7), (9, 4, 3))
        self.assertEqual(extract_target("1-2__base__embedding.dat", "DROZY", kss_scores=scores), 5.0)
        self.assertEqual(extract_target("2-1__base__embedding.dat", "DROZY", kss_scores=scores), 9.0)

    def test_bad_ids(self):
        scores = ((1, 2),)
        for video in ("0-1__base__.dat", "1-3__base__.dat", "invalid__base__.dat"):
            with self.subTest(video=video), self.assertRaises(ValueError):
                extract_target(video, "DROZY", kss_scores=scores)
        with self.assertRaises(ValueError):
            extract_target("invalid", "UTA")
        with self.assertRaises(ValueError):
            extract_target("1-1__base__.dat", "DROZY")

    def test_cached_kss_table(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "KSS.txt"
            path.write_text("1  5 7\n9\t4 3\n", encoding="utf-8")
            first = load_kss_scores(str(path))
            self.assertEqual(first, ((1, 5, 7), (9, 4, 3)))
            self.assertIs(first, load_kss_scores(str(path)))


if __name__ == "__main__":
    unittest.main()
