# Technical Notes

## One-minute walkthrough

The practical problem was that UAV thermal and RGB streams are complementary but not geometrically stable enough to fuse with one fixed projection. I built the workflow around that constraint: timestamp-aligned frame extraction, independent per-sensor labels, automated checks before model training, and one tuned YOLOv8 model per modality. The final Thermal and RGB models achieved 84.70% and 85.93% validation mAP@50 respectively. I also tested transfer on a 109-frame unseen winter sequence and reported 97% human-count coverage carefully as a sequence-level proxy, not recall.

## Questions this project is ready to answer

- Why not use one multimodal detector? The geometry and data volume did not support a reliable learned fusion experiment; independent streams provided an auditable, stable baseline.
- Why does Thermal use 640 px while RGB uses 1280 px? The resolution sweep showed different optima: upscaling the lower-resolution thermal stream did not help, while the RGB stream benefited from higher spatial detail.
- What did validation protect against? It checked label structure, class IDs, normalized boxes, image/label pairing, class balance, and the presence of negative frames before training.
- What would make the evaluation stronger? A held-out test set with one-to-one OOD matching, domain-specific canopy data, and tracking-aware metrics.

## Experiment evidence

### Resolution sweep

| Modality | 640 px | 960 px | 1280 px | Selected |
| --- | ---: | ---: | ---: | --- |
| Thermal | **84.70%** | 80.10% | 83.00% | 640 px |
| RGB | 79.90% | 84.80% | **85.93%** | 1280 px |

Upscaling the lower-resolution Thermal stream did not improve validation mAP@50. RGB benefited from the additional spatial detail, which is especially useful for small aerial targets.

### Optimizer sweep at 640 px

| Modality | SGD | AdamW | Interpretation |
| --- | ---: | ---: | --- |
| Thermal | **84.70%** | 80.36% | SGD remained stronger |
| RGB | 79.90% | **81.88%** | AdamW improved the 640 px baseline |

The final RGB run still used SGD at 1280 px because its 85.93% mAP@50 exceeded the 640 px AdamW result. Resolution and optimizer choices therefore cannot be treated as independent global rules.

## Metric boundaries

- Validation mAP@50 is reported from the final 80/20 split, not a separate hold-out test set.
- OOD count coverage is `detections / annotations` over a sequence. It is useful as a transfer signal but does not measure one-to-one object recall.
- F1-derived confidence thresholds are operating points, not universal constants; a deployment domain shift requires recalibration.
