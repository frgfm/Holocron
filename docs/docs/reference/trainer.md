# holocron.trainer

`holocron.trainer` provides some basic objects for training purposes.

Pass `device="cpu"`, `device="cuda:0"`, or `device="mps"` to a trainer. Use
`device="auto"` to prefer CUDA, then Apple Silicon MPS, then CPU. The trainer
moves the model, criterion, input tensors, and targets to that device.

Existing `gpu=0` callers still select CUDA. Omitting both `gpu` and `device`
retains the library's CPU default. Specify only one of these arguments.
`to_cuda` remains a compatibility alias for `to_device`.

With `amp=True`, CPU uses BF16 autocasting without gradient scaling; CUDA and
MPS use FP16 autocasting with gradient scaling. Without AMP, training stays in
the model's original precision. MPS mixed precision was validated with the
locked PyTorch 2.13 environment on an M3 Pro; backend operator support can vary
with PyTorch and macOS versions.

::: holocron.trainer.Trainer
    options:
        heading_level: 3
        show_object_full_path: false
        members:
            - set_device
            - to_cuda
            - to_device
            - save
            - load
            - fit_n_epochs
            - find_lr
            - plot_recorder
            - check_setup


## Image classification

::: holocron.trainer
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - ClassificationTrainer
            - BinaryClassificationTrainer

## Semantic segmentation

::: holocron.trainer
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - SegmentationTrainer

## Object detection

::: holocron.trainer
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - DetectionTrainer

## Miscellaneous

::: holocron.trainer
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - freeze_bn
            - freeze_model
            - resolve_device
