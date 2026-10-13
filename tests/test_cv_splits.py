"""Synthetic group-split checks that do not require research datasets or PyTorch."""

import unittest

from Zh.src.utils.cv_splits import (
    validate_group_kfold_input,
    validate_group_partition,
)


class GroupSplitTests(unittest.TestCase):
    def setUp(self):
        self.samples = ("A_1", "B_1", "A_2", "B_2", "A_3", "B_3")
        self.group_of = lambda sample: sample.split("_")[1]

    def test_valid_fold_input(self):
        validate_group_kfold_input(self.samples, 3, self.group_of)

    def test_valid_group_partition(self):
        validate_group_partition(
            self.samples,
            ("A_1", "B_1", "A_2", "B_2"),
            ("A_3", "B_3"),
            self.group_of,
        )

    def test_rejects_invalid_fold_count(self):
        for count in (0, 1, 4):
            with self.subTest(count=count), self.assertRaises(ValueError):
                validate_group_kfold_input(self.samples, count, self.group_of)

    def test_rejects_duplicate_input_ids(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_group_kfold_input(("A_1", "A_1", "B_2"), 2, self.group_of)

    def test_rejects_empty_input(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            validate_group_kfold_input([], 2, self.group_of)

    def test_rejects_participant_overlap(self):
        with self.assertRaisesRegex(ValueError, "participant groups"):
            validate_group_partition(
                self.samples,
                ("A_1", "A_2", "B_2", "B_3"),
                ("B_1", "A_3"),
                self.group_of,
            )

    def test_rejects_sample_overlap(self):
        with self.assertRaisesRegex(ValueError, "share samples"):
            validate_group_partition(
                ("A_1", "B_2"),
                ("A_1",),
                ("A_1", "B_2"),
                self.group_of,
            )

    def test_rejects_missing_samples(self):
        with self.assertRaisesRegex(ValueError, "cover"):
            validate_group_partition(
                self.samples,
                ("A_1", "B_1"),
                ("A_2", "B_2"),
                self.group_of,
            )

    def test_rejects_empty_partition(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            validate_group_partition(
                self.samples, self.samples, (), self.group_of
            )

    def test_rejects_duplicate_partition_samples(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_group_partition(
                ("A_1", "B_2"),
                ("A_1", "A_1"),
                ("B_2",),
                self.group_of,
            )

    def test_group_extractor_can_be_injected(self):
        items = (101, 102, 201, 202)
        group_of = lambda value: value // 100
        validate_group_kfold_input(items, 2, group_of)
        validate_group_partition(items, (101, 102), (201, 202), group_of)


if __name__ == "__main__":
    unittest.main()
