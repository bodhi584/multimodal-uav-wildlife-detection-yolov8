"""Train one modality-specific YOLOv8 detector for UAV wildlife imagery."""

from __future__ import annotations

import argparse
from pathlib import Path

FINAL_CONFIGS = {
    "thermal": {"imgsz": 640, "confidence": 0.563},
    "rgb": {"imgsz": 1280, "confidence": 0.350},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--modality", choices=FINAL_CONFIGS, required=True)
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, help="Override the reported final image size")
    parser.add_argument("--optimizer", default="SGD", choices=["SGD", "AdamW"])
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"YOLO dataset configuration not found: {args.data}")

    from ultralytics import YOLO

    imgsz = args.imgsz or FINAL_CONFIGS[args.modality]["imgsz"]
    run_name = f"{args.modality}_yolov8n_{imgsz}px_{args.optimizer.lower()}"

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=imgsz,
        batch=args.batch,
        optimizer=args.optimizer,
        patience=20,
        project=str(args.project),
        name=run_name,
        save=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
