# holocron.models

The models subpackage contains definitions of models for addressing
different tasks, including: image classification, pixelwise semantic
segmentation, object detection, and text recognition.

## Discover models

List model factory functions and create a model without indexing Python modules:

```python
from holocron.models import get_model, get_model_info, list_checkpoints, list_models

names = list_models(task="classification", pretrained=True)
info = get_model_info("convnext_atto")
checkpoints = list_checkpoints(info.name)
model = get_model(info.name, pretrained=info.pretrained, checkpoint=checkpoints[0] if checkpoints else None)
```

`pretrained` reports built-in weight availability, including legacy weights. It
does not establish benchmark quality. `list_checkpoints` returns existing typed
metadata and stays empty for legacy weights. Factory arguments pass through unchanged;
for example, `get_model("yolo26n", num_classes=20)`. Recognition currently exposes
classes rather than factory functions, so its catalog list is empty. Discovery
does not instantiate models or download weights.

Some detection and segmentation factories load pretrained backbones by default.
`pretrained=False` disables full-model weights only. Pass
`pretrained_backbone=False` to those factories to avoid backbone downloads.

## Local inference measurements

Benchmark every public model factory plus `CharacterClassifier` and `CTCRecognizer`:

```bash
UV_TORCH_BACKEND=cpu uv sync --locked --extra scripts
uv run --no-sync python scripts/eval_latency.py all --output /tmp/holocron-inference.json
```

Replace `all` with a model name, or use `--list-models` to inspect coverage.
The Make target accepts `LATENCY_ARCH=all` and `LATENCY_ARGS="--output /tmp/holocron-inference.json"`.
Defaults are 100 iterations, 10 warmups and five fresh processes per model, run
sequentially. Add `--it 1 --warmup 0 --repeat 1` for a coverage smoke run.

Models use FP32 evaluation forwards under `torch.inference_mode()`, with seeded
random weights/inputs and no backbone downloads. RepVGG, MobileOne and RepViT are
reparametrized; other models use ordinary evaluation forwards, including detection
postprocessing. Preprocessing and recognition decoding are excluded.

Inputs default to `(batch, 3, 224, 224)`, except YOLOv1 requires 448 pixels and
recognition uses `(batch, 1, 32, 128)` with full-width lengths and 83 characters.
`--size` overrides the image side or recognition width; omit it for the full suite.
Use `--help` for class, batch, device, seed and thread settings (CPU threads default
to one). Use `--device cuda:0` or `--device mps` for GPUs; GPU timing waits for completion.
`--backend onnx` measures ONNX Runtime on CPU for one classification model, after checking its output against PyTorch.
Summaries report first-call, median and p95 latency, and throughput, as medians across workers.

JSON records contain UTC time, Git revision/dirty state, CPU hardware/affinity,
OS/runtime versions, tested forward/source path, input settings, latency, throughput
and peak RSS in MiB, with raw repeats. Latency excludes setup and transfer; RSS
includes imports and model construction and is unavailable on Windows. Fresh
processes isolate memory peaks. Progress is saved with `complete: false` until
every model succeeds; errors exit nonzero. These are synthetic performance checks;
compare runs on the same hardware, OS, runtime versions and input settings.
To compare dependency versions, use one `uv sync --locked` environment per version, then
alternate runs with identical settings; run the baseline twice first to gauge noise.

## Support status

| Task | Architectures | Published checkpoints | Training | ONNX | Status |
|---|---|---|---|---|---|
| Classification | 15 families | Imagenette checkpoints with top-1/top-5 metrics; selected ReXNet ImageNet-1K checkpoints | [Reference script](https://github.com/frgfm/holocron/blob/main/references/classification/train.py) | [CPU inference parity](#onnx-export) tested for all factories | **Validated** |
| Semantic segmentation | U-Net, U-Net++, UNet3+, [YOLO26 nano](segmentation/yolo26.md) | Only the legacy `unet_rexnet13` weights; dataset and metric are not documented | [Reference script](https://github.com/frgfm/holocron/blob/main/references/segmentation/train.py) | [CPU inference parity](#onnx-export) tested for all factories | **Unbenchmarked** on standard benchmarks; YOLO26 small-data learning checks available |
| Object detection | YOLOv1, YOLOv2, YOLOv4, YOLO26 nano | None | [Reference script](https://github.com/frgfm/holocron/blob/main/references/detection/train.py) | [CPU inference parity](#onnx-export) tested for all factories | **Experimental**; [YOLOv4 learning check verified](#yolo-training-validation) |
| [Text recognition](recognition.md) | Shared glyph CNN, BiGRU/CTC | None; training produces local checkpoints | [Synthetic curriculum](https://github.com/frgfm/holocron/tree/main/references/recognition) | Not validated | **Experimental**; synthetic CPU benchmark |

**Validated** means published task checkpoints and metrics are available.
**Unbenchmarked** means an implementation or legacy weight exists without a
documented evaluation dataset and metric. **Experimental** means the API is
available, but published task weights and a confirmed benchmark are not.

## ONNX export

From a repository checkout, export a trained model and check its answers in ONNX Runtime:

```bash
uv sync --locked --extra scripts
uv run --no-sync python scripts/export_to_onnx.py resnet18 \
    --checkpoint checkpoints/model.pth --num-classes 10 --height 224 --width 224 --path model.onnx
```

The checkpoint can be a raw state dictionary or a trainer checkpoint containing `model`.
Use the same architecture, class count and input channels as training. Export loads the
training weights without downloading backbone weights.

The contract is CPU FP32, ONNX opset 20, and a fixed `(batch, channels, height, width)` input
named `images`. Re-export for a different shape. Classification and segmentation return `logits`;
detection returns `boxes_0`, `scores_0`, `labels_0`, then the same fields for each subsequent
batch item. Boxes are normalized xyxy coordinates; detection counts can vary with the input.
Provide the same resized, normalized image tensors used for PyTorch inference; preprocessing
and class names are not bundled into the graph.
YOLOv1 requires 448-by-448 images; YOLO26 detection requires dimensions divisible by 32.

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
images = np.load("preprocessed_images.npy").astype(np.float32)  # shape (1, 3, 224, 224)
outputs = dict(zip((output.name for output in session.get_outputs()), session.run(None, {"images": images})))
```

The exporter checks the ONNX graph and compares runtime shapes, dtypes and finite values with
PyTorch on the sample, blank and fresh random inputs. Float outputs use `rtol=1e-3, atol=3e-5`;
labels must match exactly. Detection records are compared independent of their order.
Near-tied class scores or equal-score top-k/NMS can change labels or selected boxes across
runtimes; such differences fail verification. These checks establish numerical parity for the
tested inputs, not dataset accuracy. The output file is replaced only after verification passes.
Dynamic shapes, quantization and Core ML conversion remain future work.

For a custom architecture, call `export_model(model, images, path)` from
`scripts.export_to_onnx.py` with an evaluation model on CPU and FP32 inputs. It accepts the same
tensor or detection-list outputs as the built-in models and raises on incompatible export or
runtime differences. The helper puts the model in evaluation mode; it leaves its inference
branches intact, so the same model can be exported again. For a smaller deployment graph,
optionally call the architecture's `reparametrize()` or `fuse()` method first. YOLO26 detection
uses its independent `to_deploy()` copy automatically.
CI discovers vision factories through `list_models()` and exercises every factory with three
classes and fixed inputs (64 pixels, or 448 for YOLOv1). Fixtures with nearly constant outputs
get up to two blank/random BatchNorm calibration batches. Each fixture must respond to its input
beyond the comparison tolerance; regressions also exercise deliberately broken exports that ignore
the image.


## Classification

Classification models expect a 4D image tensor as an input (N x C x H x W) and returns a 2D output (N x K).
The output represents the classification scores for each output classes.

### Supported architectures
* [ResNet](./classification/resnet.md)
* [ResNeXt](./classification/resnext.md)
* [Res2Net](./classification/res2net.md)
* [TridentNet](./classification/tridentnet.md)
* [ConvNeXt](./classification/convnext.md)
* [PyConvResNet](./classification/pyconv_resnet.md)
* [ReXNet](./classification/rexnet.md)
* [SKNet](./classification/sknet.md)
* [DarkNet](./classification/darknet.md)
* [DarkNetV2](./classification/darknetv2.md)
* [DarkNetV3](./classification/darknetv3.md)
* [DarkNetV4](./classification/darknetv4.md)
* [RepVGG](./classification/repvgg.md)
* [MobileOne](./classification/mobileone.md)
* [RepViT](./classification/repvit.md)

### Available checkpoints

Most classification checkpoints were trained on [Imagenette](https://github.com/fastai/imagenette),
a ten-class subset of ImageNet. Selected ReXNet variants also provide ImageNet-1K
weights with 1,000 output classes. Choose a checkpoint by its dataset, not just
its architecture name. Accuracy values from these two datasets are not comparable.

Pass a `Checkpoint` object, for example
`rexnet1_0x(checkpoint=ReXNet1_0x_Checkpoint.IMAGENETTE.value)`, to select weights
explicitly. For models with checkpoint enums, `pretrained=True`
selects `DEFAULT.value`; it does not always select Imagenette. The checkpoint
defines the preprocessing (`checkpoint.transforms()`), output categories and evaluation metrics. See the
[quick start](../../index.md#quick-start) for inference and the
[transfer-learning guide](../../getting-started/classification.md) to replace the
classifier after loading weights.

Weights may be trained in Holocron or ported from another implementation. A
matching architecture name does not make a torchvision or `timm` state dictionary
compatible: parameter names, shapes and preprocessing may differ. Use the
published Holocron checkpoint or a documented adapter, such as
[RepViT's official checkpoint import](classification/repvit.md#use-the-authors-checkpoint).

An implemented architecture does not guarantee pretrained weights. When no
weights are available, legacy weight loaders log
`Invalid model URL, using default initialization.` and keep the model's initial
parameters. The YOLO26 detection and segmentation builders instead raise
`ValueError` for `pretrained=True`. Check the model's checkpoint documentation
before using it for inference; train models without weights with the
[reference scripts](https://github.com/frgfm/Holocron/tree/main/references).

The table below lists classification checkpoints with recorded metrics.
`darknet24` and `tridentnet50` also have legacy Imagenette weights, but no
recorded evaluation metrics. Their `model.default_cfg` is a dictionary with
`input_shape`, `mean`, `std` and `classes` keys, rather than a `Checkpoint` object.

CPU and MPS: median batch-1 latency at 224×224 on an Apple M3 Pro (one CPU thread, or its GPU),
with PyTorch 2.13, from [`eval_latency.py`](#local-inference-measurements).

The chart compares only the 27 Imagenette checkpoints. ImageNet-1K rows remain
in the table for reference but use a different evaluation dataset.

![Scatter plot of Imagenette top-one accuracy against parameter count, with the Pareto frontier and default ResNet-18 checkpoint highlighted.](../../img/checkpoint-accuracy-vs-parameters.svg)

| **Checkpoint** | **Acc@1** | **Acc@5** | **Params** | **Size (MB)** | **CPU (ms)** | **MPS (ms)** |
|---|---|---|---|---|---|---|
| [`CSPDarknet53_Checkpoint.IMAGENETTE`][holocron.models.CSPDarknet53_Checkpoint.IMAGENETTE] | 94.50% | 99.64% | 26.6M | 101.8 | 34.7 | 8.5 |
| [`CSPDarknet53_Mish_Checkpoint.IMAGENETTE`][holocron.models.CSPDarknet53_Mish_Checkpoint.IMAGENETTE] | 94.65% | 99.69% | 26.6M | 101.8 | 136.4 | 9.3 |
| [`ConvNeXt_Atto_Checkpoint.IMAGENETTE`][holocron.models.ConvNeXt_Atto_Checkpoint.IMAGENETTE] | 87.59% | 98.32% | 3.4M | 12.9 | 131.4 | 2.1 |
| [`Darknet19_Checkpoint.IMAGENETTE`][holocron.models.Darknet19_Checkpoint.IMAGENETTE] | 93.86% | 99.36% | 19.8M | 75.7 | 19.2 | 3.1 |
| [`Darknet53_Checkpoint.IMAGENETTE`][holocron.models.Darknet53_Checkpoint.IMAGENETTE] | 94.17% | 99.57% | 40.6M | 155.1 | 35.8 | 8.7 |
| [`MobileOne_S0_Checkpoint.IMAGENETTE`][holocron.models.MobileOne_S0_Checkpoint.IMAGENETTE] | 88.08% | 98.83% | 4.3M | 16.9 | 6.7 | 1.7 |
| [`MobileOne_S1_Checkpoint.IMAGENETTE`][holocron.models.MobileOne_S1_Checkpoint.IMAGENETTE] | 91.26% | 99.18% | 3.6M | 13.9 | 11.8 | 1.9 |
| [`MobileOne_S2_Checkpoint.IMAGENETTE`][holocron.models.MobileOne_S2_Checkpoint.IMAGENETTE] | 91.31% | 99.21% | 5.9M | 22.8 | 14.4 | 2.4 |
| [`MobileOne_S3_Checkpoint.IMAGENETTE`][holocron.models.MobileOne_S3_Checkpoint.IMAGENETTE] | 91.06% | 99.31% | 8.1M | 31.5 | 17.8 | 3.0 |
| [`ReXNet1_0x_Checkpoint.IMAGENET1K`][holocron.models.ReXNet1_0x_Checkpoint.IMAGENET1K] | 77.86% | 93.87% | 4.8M | 13.7 | 19.8 | 6.5 |
| [`ReXNet1_0x_Checkpoint.IMAGENETTE`][holocron.models.ReXNet1_0x_Checkpoint.IMAGENETTE] | 94.39% | 99.62% | 3.5M | 13.7 | 19.8 | 6.5 |
| [`ReXNet1_3x_Checkpoint.IMAGENET1K`][holocron.models.ReXNet1_3x_Checkpoint.IMAGENET1K] | 79.50% | 94.68% | 7.6M | 13.7 | 26.1 | 8.2 |
| [`ReXNet1_3x_Checkpoint.IMAGENETTE`][holocron.models.ReXNet1_3x_Checkpoint.IMAGENETTE] | 94.88% | 99.39% | 5.9M | 22.8 | 26.1 | 8.2 |
| [`ReXNet1_5x_Checkpoint.IMAGENET1K`][holocron.models.ReXNet1_5x_Checkpoint.IMAGENET1K] | 80.31% | 95.17% | 9.7M | 13.7 | 30.3 | 6.6 |
| [`ReXNet1_5x_Checkpoint.IMAGENETTE`][holocron.models.ReXNet1_5x_Checkpoint.IMAGENETTE] | 94.47% | 99.62% | 7.8M | 30.2 | 30.3 | 6.6 |
| [`ReXNet2_0x_Checkpoint.IMAGENET1K`][holocron.models.ReXNet2_0x_Checkpoint.IMAGENET1K] | 80.31% | 95.17% | 16.4M | 13.7 | 39.5 | 7.7 |
| [`ReXNet2_0x_Checkpoint.IMAGENETTE`][holocron.models.ReXNet2_0x_Checkpoint.IMAGENETTE] | 95.24% | 99.57% | 13.8M | 53.1 | 39.5 | 7.7 |
| [`ReXNet2_2x_Checkpoint.IMAGENETTE`][holocron.models.ReXNet2_2x_Checkpoint.IMAGENETTE] | 95.44% | 99.46% | 16.7M | 64.1 | 43.8 | 8.6 |
| [`RepVGG_A0_Checkpoint.IMAGENETTE`][holocron.models.RepVGG_A0_Checkpoint.IMAGENETTE] | 92.92% | 99.46% | 24.7M | 94.6 | 17.8 | 3.5 |
| [`RepVGG_A1_Checkpoint.IMAGENETTE`][holocron.models.RepVGG_A1_Checkpoint.IMAGENETTE] | 93.78% | 99.18% | 30.1M | 115.1 | 24.2 | 4.6 |
| [`RepVGG_A2_Checkpoint.IMAGENETTE`][holocron.models.RepVGG_A2_Checkpoint.IMAGENETTE] | 93.63% | 99.39% | 48.6M | 185.8 | 35.8 | 7.9 |
| [`RepVGG_B0_Checkpoint.IMAGENETTE`][holocron.models.RepVGG_B0_Checkpoint.IMAGENETTE] | 92.69% | 99.21% | 31.8M | 121.8 | 28.6 | 5.0 |
| [`RepVGG_B1_Checkpoint.IMAGENETTE`][holocron.models.RepVGG_B1_Checkpoint.IMAGENETTE] | 93.96% | 99.39% | 100.8M | 385.1 | 62.4 | 13.6 |
| [`RepVGG_B2_Checkpoint.IMAGENETTE`][holocron.models.RepVGG_B2_Checkpoint.IMAGENETTE] | 94.14% | 99.57% | 157.5M | 601.2 | 88.1 | 19.5 |
| [`Res2Net50_26w_4s_Checkpoint.IMAGENETTE`][holocron.models.Res2Net50_26w_4s_Checkpoint.IMAGENETTE] | 93.94% | 99.41% | 23.7M | 90.6 | 29.1 | 9.2 |
| [`ResNeXt50_32x4d_Checkpoint.IMAGENETTE`][holocron.models.ResNeXt50_32x4d_Checkpoint.IMAGENETTE] | 94.55% | 99.49% | 23.0M | 88.1 | 33.2 | 7.4 |
| [`ResNet18_Checkpoint.IMAGENETTE`][holocron.models.ResNet18_Checkpoint.IMAGENETTE] | 93.61% | 99.46% | 11.2M | 42.7 | 16.4 | 2.7 |
| [`ResNet34_Checkpoint.IMAGENETTE`][holocron.models.ResNet34_Checkpoint.IMAGENETTE] | 93.81% | 99.49% | 21.3M | 81.3 | 24.9 | 4.4 |
| [`ResNet50D_Checkpoint.IMAGENETTE`][holocron.models.ResNet50D_Checkpoint.IMAGENETTE] | 94.65% | 99.52% | 23.5M | 90.1 | 28.4 | 7.6 |
| [`ResNet50_Checkpoint.IMAGENETTE`][holocron.models.ResNet50_Checkpoint.IMAGENETTE] | 93.78% | 99.54% | 23.5M | 90 | 26.7 | 7.1 |
| [`SKNet50_Checkpoint.IMAGENETTE`][holocron.models.SKNet50_Checkpoint.IMAGENETTE] | 94.37% | 99.54% | 35.2M | 134.7 | 46.9 | 36.1 |




## Object Detection

!!! warning "Experimental"
    YOLOv4's corrected implementation passes regression tests and a CPU
    fixed-batch learning check. Full CUDA/VOC training on the repaired
    implementation remains pending. No pretrained detection checkpoints or
    paper-level accuracy results are published.

Object detection models expect a 4D image tensor as an input (N x C x H x W) and returns a list of dictionaries.
In evaluation mode, each dictionary has three keys: `boxes` (normalized xmin, ymin, xmax, ymax coordinates),
`scores` (objectness multiplied by the top class probability for YOLOv1/v2/v4;
class probability for YOLO26), and `labels` (class indices).
In training mode, pass a list of target dictionaries with normalized `boxes` and integer `labels`;
the model returns a loss dictionary.

```python
import holocron.models as models

yolov2 = models.yolov2(num_classes=10)
```

### YOLO training validation

YOLOv4 uses a CSPDarknet53-Mish backbone with an SPP/PAN neck and three detection scales.
Its corrected training uses global and additional anchor matches, constant objectness BCE targets,
summed classification BCE, and area-weighted differentiable CIoU. Box geometry is computed in
FP32 under AMP, and every scale is decoded before combined-confidence filtering and class-aware NMS.

The default `yolov4(num_classes=20)` loads a pretrained **Imagenette backbone**, even without
`pretrained=True`. The detector head starts from scratch. The reference script's
`--freeze-until backbone` freezes the pretrained weights; `--pretrained` should not be used to
request unavailable detection weights.

The [CPU learning diagnostic](https://github.com/frgfm/holocron/blob/main/references/detection/check_yolov4.py)
trains the actual neck and head for 500 SGD updates on two synthetic 96-pixel rectangle images,
repeated to a batch of eight. It caches the frozen backbone features and disables DropBlock
only for this memorization check. Normal evaluation and default NMS give one correctly labeled
detection per image, with IoUs **0.8113 / 0.7945**, no false positives, and finite losses and
parameter gradients. The pretrained backbone weights remain unchanged, and a trainer checkpoint is saved.

| Validation | Result |
| --- | --- |
| YOLOv4 fixed-batch learning | Verified; detection error 99.43% with the old initialization restored, versus 0% with corrected initialization |
| YOLOv1/v2 losses and shared inference | Loss behavior preserved; regression coverage for combined confidence and class-aware NMS |
| YOLOv1/v2/v4 ONNX | Fixed-shape export and ONNX Runtime CPU parity tested; accuracy not benchmarked |
| Full CUDA/VOC training of the repaired YOLOv4 | Pending; synthetic results do not establish validation-set accuracy |

YOLOv3 is not implemented as a detector. Its Darknet-53 classification backbone is available.
The [detection guide](https://github.com/frgfm/holocron/blob/main/references/detection/README.md)
contains the matched-control results, reproducible commands, learning-rate finder options,
metric definitions, and the five-epoch CUDA/VOC acceptance gate.

### YOLO family

[YOLO26 nano](detection/yolo26.md) adds dual-head training and NMS-free inference,
with 2.41M parameters in its fused 80-class inference model. Its full published
training recipe and pretrained weights are not included.

::: holocron.models.detection.yolo.YOLOv1
    options:
        heading_level: 4
        members: no
        show_bases: false

::: holocron.models.detection.yolo.yolov1
    options:
        heading_level: 4

::: holocron.models.detection.yolov2.YOLOv2
    options:
        heading_level: 4
        members: no
        show_bases: false

::: holocron.models.detection.yolov2.yolov2
    options:
        heading_level: 4

::: holocron.models.detection.yolov4.YOLOv4
    options:
        heading_level: 4
        members: no
        show_bases: false

::: holocron.models.detection.yolov4.yolov4
    options:
        heading_level: 4


## Semantic Segmentation

!!! note "Unbenchmarked"
    `unet_rexnet13` is the only segmentation model with a legacy checkpoint.
    Its training dataset and evaluation metric are not documented. Other
    segmentation architectures have no published weights.

Semantic segmentation models expect a 4D image tensor as an input (N x C x H x W) and returns a classification score
tensor of size (N x K x Ho x Wo).

```python
import holocron.models as models

unet = models.unet(num_classes=10)
```

### U-Net family

::: holocron.models.segmentation.unet.UNet
    options:
        heading_level: 4
        members: no
        show_bases: false

::: holocron.models.segmentation.unet.unet
    options:
        heading_level: 4

::: holocron.models.segmentation.unet.DynamicUNet
    options:
        heading_level: 4
        members: no
        show_bases: false

::: holocron.models.segmentation.unet.unet2
    options:
        heading_level: 4

::: holocron.models.segmentation.unet.unet_tvvgg11
    options:
        heading_level: 4

::: holocron.models.segmentation.unet.unet_tvresnet34
    options:
        heading_level: 4

::: holocron.models.segmentation.unet.unet_rexnet13
    options:
        heading_level: 4


::: holocron.models.segmentation.unetpp.UNetp
    options:
        heading_level: 4
        members: no
        show_bases: false

::: holocron.models.segmentation.unetpp.unetp
    options:
        heading_level: 4

::: holocron.models.segmentation.unetpp.UNetpp
    options:
        heading_level: 4
        members: no
        show_bases: false

::: holocron.models.segmentation.unetpp.unetpp
    options:
        heading_level: 4

::: holocron.models.segmentation.unet3p.UNet3p
    options:
        heading_level: 4
        members: no
        show_bases: false

::: holocron.models.segmentation.unet3p.unet3p
    options:
        heading_level: 4

### YOLO26 family

See [YOLO26 nano semantic segmentation](segmentation/yolo26.md) for the model
API, training output, parameter counts, and deployment limits.
