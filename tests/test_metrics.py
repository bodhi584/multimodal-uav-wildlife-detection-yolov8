"""Tests for the public evaluation helpers."""

import unittest

from src.metrics import count_coverage


class CountCoverageTests(unittest.TestCase):
    def test_reported_ood_result(self) -> None:
        self.assertAlmostEqual(count_coverage(555, 572), 0.9702797203)

    def test_over_count_is_not_clipped(self) -> None:
        self.assertEqual(count_coverage(11, 10), 1.1)

    def test_annotations_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            count_coverage(0, 0)

    def test_detections_must_be_non_negative(self) -> None:
        with self.assertRaises(ValueError):
            count_coverage(-1, 10)


if __name__ == "__main__":
    unittest.main()
