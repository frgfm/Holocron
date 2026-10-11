# Changelog

## Unreleased

- Support CPU, CUDA, and Apple Silicon MPS training. Reference recipes now default to `--device auto`; use `--device cpu` to retain the previous CPU default.

## v0.3.0 (2026-10-03)
Release note: [v0.3.0](https://github.com/frgfm/Holocron/releases/tag/v0.3.0)

- Require Python 3.11 or newer and update the PyTorch, torchvision, Hugging Face Hub, and Pillow requirements.
- Add ConvNeXt, MobileOne, RepViT, character classifiers, and CTC text recognition, with expanded classification checkpoints and reference training tools.
- Add Adan and AdEMAMix, gradient accumulation, dataset analysis, and ONNX export and latency utilities.
- Secure remote checkpoint loading and restore UNet-family segmentation training and RepViT deployment conversion correctness.
- Repair YOLOv4 target assignment, BCE/CIoU losses, prediction initialization, AMP box geometry, and custom-anchor checkpoint loading.
- Correct shared confidence filtering, class-aware NMS, and detection evaluation matching while preserving YOLOv1/v2 loss behavior.
- Add a reproducible YOLOv4 CPU learning diagnostic with a frozen pretrained backbone, finite-gradient checks, normal-inference acceptance, and a saved checkpoint. Full CUDA/VOC training of the repaired implementation remains pending; YOLOv3 remains unimplemented as a detector.
- Expose detection LR finder bounds and progress, plot the complete recorded range, and make CodeCarbon quiet by default with `--verbose-codecarbon` available.

## v0.2.1 (2022-07-16)
Release note: [v0.2.1](https://github.com/frgfm/Holocron/releases/tag/v0.2.1)

## v0.2.0 (2022-02-05)
Release note: [v0.2.0](https://github.com/frgfm/Holocron/releases/tag/v0.2.0)

## v0.1.3 (2020-10-27)
Release note: [v0.1.3](https://github.com/frgfm/Holocron/releases/tag/v0.1.3)

## v0.1.2 (2020-06-21)
Release note: [v0.1.2](https://github.com/frgfm/Holocron/releases/tag/v0.1.2)

## v0.1.1 (2020-05-12)
Release note: [v0.1.1](https://github.com/frgfm/Holocron/releases/tag/v0.1.1)

## v0.1.0 (2020-05-11)
Release note: [v0.1.0](https://github.com/frgfm/Holocron/releases/tag/v0.1.0)
