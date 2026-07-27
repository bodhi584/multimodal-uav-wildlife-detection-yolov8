"""Tests for YOLO label validation."""

import tempfile
import unittest
from pathlib import Path

from src.validate_dataset import validate_label


class LabelValidationTests(unittest.TestCase):
    def write_label(self, content: str) -> Path:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        label_path = Path(temporary_directory.name) / "sample.txt"
        label_path.write_text(content)
        return label_path

    def test_valid_label(self) -> None:
        counts = validate_label(self.write_label("2 0.5 0.5 0.2 0.4\n"), class_count=6)
        self.assertEqual(counts[2], 1)

    def test_empty_negative_label(self) -> None:
        self.assertFalse(validate_label(self.write_label(""), class_count=6))

    def test_rejects_box_outside_image(self) -> None:
        with self.assertRaises(ValueError):
            validate_label(self.write_label("0 0.95 0.5 0.2 0.2\n"), class_count=6)

    def test_rejects_unknown_class(self) -> None:
        with self.assertRaises(ValueError):
            validate_label(self.write_label("6 0.5 0.5 0.2 0.2\n"), class_count=6)


if __name__ == "__main__":
    unittest.main()
