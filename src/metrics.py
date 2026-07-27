"""Small evaluation utilities used by the public project summary."""

from __future__ import annotations

import argparse


def count_coverage(detections: int, annotations: int) -> float:
    """Return a sequence-level detection-count ratio.

    This value is not object-level recall because it does not match detections
    to annotations. Values above 1.0 are possible when detections over-count.
    """
    if detections < 0:
        raise ValueError("detections must be non-negative")
    if annotations <= 0:
        raise ValueError("annotations must be positive")
    return detections / annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=int, required=True)
    parser.add_argument("--annotations", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coverage = count_coverage(args.detections, args.annotations)
    print(f"Count coverage: {coverage:.4f} ({coverage:.2%})")
    print("Interpretation: sequence-level count proxy, not object-level recall")


if __name__ == "__main__":
    main()
