"""Validate a YOLO dataset before launching a costly training run."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


CLASS_NAMES = ("Human", "Vehicle", "Deer", "Hare", "Dog", "Duck")
SPLITS = ("train", "val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="YOLO dataset root")
    parser.add_argument("--classes", type=int, default=len(CLASS_NAMES))
    return parser.parse_args()


def validate_label(label_path: Path, class_count: int) -> Counter[int]:
    counts: Counter[int] = Counter()
    for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{label_path}:{line_number} must contain 5 YOLO values")
        class_id = int(values[0])
        x_center, y_center, width, height = [float(value) for value in values[1:]]
        if not 0 <= class_id < class_count:
            raise ValueError(f"{label_path}:{line_number} has invalid class id {class_id}")
        if not 0 <= x_center <= 1 or not 0 <= y_center <= 1:
            raise ValueError(f"{label_path}:{line_number} has coordinates outside [0, 1]")
        if not 0 < width <= 1 or not 0 < height <= 1:
            raise ValueError(f"{label_path}:{line_number} has a non-positive or oversized box")
        if (
            x_center - width / 2 < 0
            or x_center + width / 2 > 1
            or y_center - height / 2 < 0
            or y_center + height / 2 > 1
        ):
            raise ValueError(f"{label_path}:{line_number} extends beyond the image boundary")
        counts[class_id] += 1
    return counts


def main() -> None:
    args = parse_args()
    total_counts: Counter[int] = Counter()

    for split in SPLITS:
        image_dir = args.dataset / "images" / split
        label_dir = args.dataset / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing images/{split} or labels/{split} under {args.dataset}")

        images = [path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        labels = list(label_dir.glob("*.txt"))
        image_stems = {image.stem for image in images}
        label_stems = {label.stem for label in labels}
        missing_labels = sorted(image_stems - label_stems)
        orphan_labels = sorted(label_stems - image_stems)
        negative_frames = sum(not label.read_text().strip() for label in labels)
        split_counts: Counter[int] = Counter()
        for label_path in labels:
            split_counts.update(validate_label(label_path, args.classes))
        total_counts.update(split_counts)

        print(f"{split}: {len(images)} images, {len(labels)} labels")
        print(f"  images without labels: {len(missing_labels)}")
        print(f"  labels without images: {len(orphan_labels)}")
        print(f"  empty labels (negative frames): {negative_frames}")
        print(f"  boxes by class: {dict(sorted(split_counts.items()))}")

        if missing_labels or orphan_labels:
            raise ValueError(f"{split} contains unmatched images or labels")

    print("total boxes by class")
    for class_id, count in sorted(total_counts.items()):
        name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
        print(f"  {class_id} ({name}): {count}")


if __name__ == "__main__":
    main()
