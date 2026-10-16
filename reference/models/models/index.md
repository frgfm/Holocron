# holocron.models

The models subpackage contains definitions of models for addressing different tasks, including: image classification, pixelwise semantic segmentation, object detection, and text recognition.

## Discover models

List model factory functions and create a model without indexing Python modules:

```python
from holocron.models import get_model, get_model_info, list_checkpoints, list_models

names = list_models(task="classification", pretrained=True)
info = get_model_info("convnext_atto")
checkpoints = list_checkpoints(info.name)
model = get_model(info.name, pretrained=info.pretrained, checkpoint=checkpoints[0] if checkpoints else None)
```

`pretrained` reports built-in weight availability, including legacy weights. It does not establish benchmark quality. `list_checkpoints` returns existing typed metadata and stays empty for legacy weights. Factory arguments pass through unchanged; for example, `get_model("yolo26n", num_classes=20)`. Recognition currently exposes classes rather than factory functions, so its catalog list is empty. Discovery does not instantiate models or download weights.

Some detection and segmentation factories load pretrained backbones by default. `pretrained=False` disables full-model weights only. Pass `pretrained_backbone=False` to those factories to avoid backbone downloads.

## Local inference measurements

Benchmark every public model factory plus `CharacterClassifier` and `CTCRecognizer`:

```bash
UV_TORCH_BACKEND=cpu uv sync --locked --extra scripts
uv run --no-sync python scripts/eval_latency.py all --output /tmp/holocron-inference.json
```

Replace `all` with a model name, or use `--list-models` to inspect coverage. The Make target accepts `LATENCY_ARCH=all` and `LATENCY_ARGS="--output /tmp/holocron-inference.json"`. Defaults are 100 iterations, 10 warmups and five fresh processes per model, run sequentially. Add `--it 1 --warmup 0 --repeat 1` for a coverage smoke run.

Models use FP32 evaluation forwards under `torch.inference_mode()`, with seeded random weights/inputs and no backbone downloads. RepVGG, MobileOne and RepViT are reparametrized; other models use ordinary evaluation forwards, including detection postprocessing. Preprocessing and recognition decoding are excluded.

Inputs default to `(batch, 3, 224, 224)`, except YOLOv1 requires 448 pixels and recognition uses `(batch, 1, 32, 128)` with full-width lengths and 83 characters. `--size` overrides the image side or recognition width; omit it for the full suite. Use `--help` for class, batch, device, seed and thread settings (CPU threads default to one). Use `--device cuda:0` or `--device mps` for GPUs; GPU timing waits for completion. `--backend onnx` measures ONNX Runtime on CPU for one classification model, after checking its output against PyTorch. Summaries report first-call, median and p95 latency, and throughput, as medians across workers.

JSON records contain UTC time, Git revision/dirty state, CPU hardware/affinity, OS/runtime versions, tested forward/source path, input settings, latency, throughput and peak RSS in MiB, with raw repeats. Latency excludes setup and transfer; RSS includes imports and model construction and is unavailable on Windows. Fresh processes isolate memory peaks. Progress is saved with `complete: false` until every model succeeds; errors exit nonzero. These are synthetic performance checks; compare runs on the same hardware, OS, runtime versions and input settings. To compare dependency versions, use one `uv sync --locked` environment per version, then alternate runs with identical settings; run the baseline twice first to gauge noise.

## Support status

| Task                                | Architectures                                                  | Published checkpoints                                                                    | Training                                                                                           | ONNX                                                          | Status                                                                                |
| ----------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Classification                      | 15 families                                                    | Imagenette checkpoints with top-1/top-5 metrics; selected ReXNet ImageNet-1K checkpoints | [Reference script](https://github.com/frgfm/holocron/blob/main/references/classification/train.py) | [CPU inference parity](#onnx-export) tested for all factories | **Validated**                                                                         |
| Semantic segmentation               | U-Net, U-Net++, UNet3+, [YOLO26 nano](../segmentation/yolo26/) | Only the legacy `unet_rexnet13` weights; dataset and metric are not documented           | [Reference script](https://github.com/frgfm/holocron/blob/main/references/segmentation/train.py)   | [CPU inference parity](#onnx-export) tested for all factories | **Unbenchmarked** on standard benchmarks; YOLO26 small-data learning checks available |
| Object detection                    | YOLOv1, YOLOv2, YOLOv4, YOLO26 nano                            | None                                                                                     | [Reference script](https://github.com/frgfm/holocron/blob/main/references/detection/train.py)      | [CPU inference parity](#onnx-export) tested for all factories | **Experimental**; [YOLOv4 learning check verified](#yolo-training-validation)         |
| [Text recognition](../recognition/) | Shared glyph CNN, BiGRU/CTC                                    | None; training produces local checkpoints                                                | [Synthetic curriculum](https://github.com/frgfm/holocron/tree/main/references/recognition)         | Not validated                                                 | **Experimental**; synthetic CPU benchmark                                             |

**Validated** means published task checkpoints and metrics are available. **Unbenchmarked** means an implementation or legacy weight exists without a documented evaluation dataset and metric. **Experimental** means the API is available, but published task weights and a confirmed benchmark are not.

## ONNX export

From a repository checkout, export a trained model and check its answers in ONNX Runtime:

```bash
uv sync --locked --extra scripts
uv run --no-sync python scripts/export_to_onnx.py resnet18 \
    --checkpoint checkpoints/model.pth --num-classes 10 --height 224 --width 224 --path model.onnx
```

The checkpoint can be a raw state dictionary or a trainer checkpoint containing `model`. Use the same architecture, class count and input channels as training. Export loads the training weights without downloading backbone weights.

The contract is CPU FP32, ONNX opset 20, and a fixed `(batch, channels, height, width)` input named `images`. Re-export for a different shape. Classification and segmentation return `logits`; detection returns `boxes_0`, `scores_0`, `labels_0`, then the same fields for each subsequent batch item. Boxes are normalized xyxy coordinates; detection counts can vary with the input. Provide the same resized, normalized image tensors used for PyTorch inference; preprocessing and class names are not bundled into the graph. YOLOv1 requires 448-by-448 images; YOLO26 detection requires dimensions divisible by 32.

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
images = np.load("preprocessed_images.npy").astype(np.float32)  # shape (1, 3, 224, 224)
outputs = dict(zip((output.name for output in session.get_outputs()), session.run(None, {"images": images})))
```

The exporter checks the ONNX graph and compares runtime shapes, dtypes and finite values with PyTorch on the sample, blank and fresh random inputs. Float outputs use `rtol=1e-3, atol=3e-5`; labels must match exactly. Detection records are compared independent of their order. Near-tied class scores or equal-score top-k/NMS can change labels or selected boxes across runtimes; such differences fail verification. These checks establish numerical parity for the tested inputs, not dataset accuracy. The output file is replaced only after verification passes. Dynamic shapes and quantization remain future work. For Apple applications, see [Core ML export](#core-ml-export).

For a custom architecture, call `export_model(model, images, path)` from `scripts.export_to_onnx.py` with an evaluation model on CPU and FP32 inputs. It accepts the same tensor or detection-list outputs as the built-in models and raises on incompatible export or runtime differences. The helper puts the model in evaluation mode; it leaves its inference branches intact, so the same model can be exported again. For a smaller deployment graph, optionally call the architecture's `reparametrize()` or `fuse()` method first. YOLO26 detection uses its independent `to_deploy()` copy automatically. CI discovers vision factories through `list_models()` and exercises every factory with three classes and fixed inputs (64 pixels, or 448 for YOLOv1). Fixtures with nearly constant outputs get up to two blank/random BatchNorm calibration batches. Each fixture must respond to its input beyond the comparison tolerance; regressions also exercise deliberately broken exports that ignore the image.

## Core ML export

Train with the [classification reference script](https://github.com/frgfm/holocron/blob/main/references/classification/train.py), or use a checkpoint you already trained. From a repository checkout on Apple Silicon:

```bash
uv sync --locked --python 3.11 --extra training --extra coreml
uv run --no-sync python references/classification/train.py ./imagenette2-320/ \
    --arch resnet18 --output-file checkpoints/model.pth
uv run --no-sync python scripts/export_to_coreml.py resnet18 \
    --checkpoint checkpoints/model.pth --num-classes 10 --height 224 --width 224 --path classifier.mlpackage
```

Raw state dictionaries and trainer checkpoints containing `model` both load strictly on CPU; use the architecture and class count from training. The CLI supports `resnet18` and `mobileone_s0`. For a MobileOne checkpoint saved after `eval().reparametrize()`, add `--reparameterized`. The reusable API also accepts custom classifiers; validate each architecture and checkpoint on the target runtime:

```python
from holocron.models.coreml import export_coreml

export_coreml(model, preprocessed_images, "classifier.mlpackage")
```

The ML Program targets **macOS 12+ / iOS 15+**, with fixed dimensions and batch one. FP32 computation is the default. By default, Apple Silicon macOS compares actual Core ML CPU predictions with PyTorch on the supplied tensor, zeros and deterministic random input (`rtol=1e-3, atol=3e-5`). It checks shapes, dtype and finite logits, preserves the caller's model, and refuses existing destinations. Linux requires `--unverified` (API: `verify=False`): tracing is checked, but Core ML inference is **not checked**. iOS deployment is intended; numerical validation is on macOS, without iOS device tests.

For FP16 computation, keep the source model and input in FP32 and select a verification runtime:

```bash
uv run --no-sync python scripts/export_to_coreml.py mobileone_s0 \
    --checkpoint checkpoints/model.pth --num-classes 10 --precision fp16 \
    --compute-units cpu_and_ne --path classifier-fp16.mlpackage
```

The API accepts the same options: `export_coreml(model, images, path, precision="fp16", compute_units="cpu_and_ne")`. FP16 keeps the `images`/`logits` interface in FP32. Tracing still uses the strict FP32 logit check; Core ML verification requires finite logits, the same top-1 class, and at most **0.005 absolute softmax probability error** on each of the three inputs. These probes do not establish dataset accuracy. Validate a trained checkpoint on representative images before deployment.

`--compute-units` accepts `cpu_only` (default), `cpu_and_gpu`, `cpu_and_ne`, and `all`. It selects the devices allowed during verification; the app chooses its own policy when loading the package. CPU+Neural Engine requires macOS 13+ (iOS 16+ in an app), and permits CPU fallback. It does not guarantee Neural Engine placement. FP32 ML Programs can run on CPU/GPU; FP16 also permits the Neural Engine. Compare the policies on the target device: `.all` can be faster than `.cpuAndNeuralEngine`. See Apple's [typed execution guide](https://apple.github.io/coremltools/docs-guides/source/typed-execution.html).

Use Python 3.11–3.13 and optional coremltools 9.0. The compatibility probe and macOS tests used Python 3.11 and PyTorch 2.14.1; coremltools warns that upstream testing ends at PyTorch 2.7.0. Tests validate the two architectures with three classes and 64×64 inputs, including fused/repeated MobileOne, FP16 `.all`, and FP16 CPU+Neural Engine. Dynamic shapes, integer quantization and broader model coverage are deferred. ONNX remains the separate cloud CPU deployment path.

Add `classifier.mlpackage` to an Xcode app target. Xcode compiles it to `classifier.mlmodelc`. The input `images` is a Float32 `MLMultiArray` in **NCHW** order `(1, 3, height, width)`; the output `logits` is `(1, num_classes)`, before softmax. Supply the same RGB resize/crop, interpolation and normalization as PyTorch evaluation. There is no image preprocessing or class label mapping inside the package. For the Imagenette reference recipe, resize the shorter edge to 232 with bilinear interpolation, then take a 224×224 center crop. Convert RGB bytes to `[0, 1]`, then subtract `(0.485, 0.456, 0.406)` and divide by `(0.229, 0.224, 0.225)` per channel. Custom training may require different values. Re-export for a different resolution.

```swift
import CoreML

// Load once, then reuse model for each prediction.
let config = MLModelConfiguration()
config.computeUnits = .cpuOnly // FP32 baseline; benchmark .all / .cpuAndNeuralEngine for FP16.
let url = Bundle.main.url(forResource: "classifier", withExtension: "mlmodelc")!
let model = try MLModel(contentsOf: url, configuration: config)

// rgb contains resized/cropped 224×224 RGB pixels, interleaved, each Float in [0, 1].
func classify(rgb: [Float], model: MLModel) throws -> MLMultiArray {
    precondition(rgb.count == 224 * 224 * 3)
    let images = try MLMultiArray(shape: [1, 3, 224, 224], dataType: .float32)
    let data = images.dataPointer.assumingMemoryBound(to: Float.self)
    let strides = images.strides.map(\.intValue)
    let mean: [Float] = [0.485, 0.456, 0.406]
    let std: [Float] = [0.229, 0.224, 0.225]
    for c in 0..<3 {
        for y in 0..<224 {
            for x in 0..<224 {
                let value = (rgb[(y * 224 + x) * 3 + c] - mean[c]) / std[c]
                data[c * strides[1] + y * strides[2] + x * strides[3]] = value
            }
        }
    }
    let input = try MLDictionaryFeatureProvider(dictionary: ["images": images])
    return try model.prediction(from: input).featureValue(for: "logits")!.multiArrayValue!
}
// The index of the largest logit selects a class in your training class order.
```

## Classification

Classification models expect a 4D image tensor as an input (N x C x H x W) and returns a 2D output (N x K). The output represents the classification scores for each output classes.

### Supported architectures

- [ResNet](../classification/resnet/)
- [ResNeXt](../classification/resnext/)
- [Res2Net](../classification/res2net/)
- [TridentNet](../classification/tridentnet/)
- [ConvNeXt](../classification/convnext/)
- [PyConvResNet](../classification/pyconv_resnet/)
- [ReXNet](../classification/rexnet/)
- [SKNet](../classification/sknet/)
- [DarkNet](../classification/darknet/)
- [DarkNetV2](../classification/darknetv2/)
- [DarkNetV3](../classification/darknetv3/)
- [DarkNetV4](../classification/darknetv4/)
- [RepVGG](../classification/repvgg/)
- [MobileOne](../classification/mobileone/)
- [RepViT](../classification/repvit/)

### Available checkpoints

Most classification checkpoints were trained on [Imagenette](https://github.com/fastai/imagenette), a ten-class subset of ImageNet. Selected ReXNet variants also provide ImageNet-1K weights with 1,000 output classes. Choose a checkpoint by its dataset, not just its architecture name. Accuracy values from these two datasets are not comparable.

Pass a `Checkpoint` object, for example `rexnet1_0x(checkpoint=ReXNet1_0x_Checkpoint.IMAGENETTE.value)`, to select weights explicitly. For models with checkpoint enums, `pretrained=True` selects `DEFAULT.value`; it does not always select Imagenette. The checkpoint defines the preprocessing (`checkpoint.transforms()`), output categories and evaluation metrics. See the [quick start](../../../#quick-start) for inference and the [transfer-learning guide](../../../getting-started/classification/) to replace the classifier after loading weights.

Weights may be trained in Holocron or ported from another implementation. A matching architecture name does not make a torchvision or `timm` state dictionary compatible: parameter names, shapes and preprocessing may differ. Use the published Holocron checkpoint or a documented adapter, such as [RepViT's official checkpoint import](../classification/repvit/#use-the-authors-checkpoint).

An implemented architecture does not guarantee pretrained weights. When no weights are available, legacy weight loaders log `Invalid model URL, using default initialization.` and keep the model's initial parameters. The YOLO26 detection and segmentation builders instead raise `ValueError` for `pretrained=True`. Check the model's checkpoint documentation before using it for inference; train models without weights with the [reference scripts](https://github.com/frgfm/Holocron/tree/main/references).

The table below lists classification checkpoints with recorded metrics. `darknet24` and `tridentnet50` also have legacy Imagenette weights, but no recorded evaluation metrics. Their `model.default_cfg` is a dictionary with `input_shape`, `mean`, `std` and `classes` keys, rather than a `Checkpoint` object.

CPU and MPS: median batch-1 latency per architecture at 224×224 on an Apple M3 Pro (one CPU thread, or its GPU), with PyTorch 2.13, from [`eval_latency.py`](#local-inference-measurements).

The chart compares only the 27 Imagenette checkpoints. ImageNet-1K rows remain in the table for reference but use a different evaluation dataset.

| **Checkpoint**                          | **Acc@1** | **Acc@5** | **Params** | **Size (MB)** | **CPU (ms)** | **MPS (ms)** |
| --------------------------------------- | --------- | --------- | ---------- | ------------- | ------------ | ------------ |
| CSPDarknet53_Checkpoint.IMAGENETTE      | 94.50%    | 99.64%    | 26.6M      | 101.8         | 34.7         | 8.5          |
| CSPDarknet53_Mish_Checkpoint.IMAGENETTE | 94.65%    | 99.69%    | 26.6M      | 101.8         | 136.4        | 9.3          |
| ConvNeXt_Atto_Checkpoint.IMAGENETTE     | 87.59%    | 98.32%    | 3.4M       | 12.9          | 131.4        | 2.1          |
| Darknet19_Checkpoint.IMAGENETTE         | 93.86%    | 99.36%    | 19.8M      | 75.7          | 19.2         | 3.1          |
| Darknet53_Checkpoint.IMAGENETTE         | 94.17%    | 99.57%    | 40.6M      | 155.1         | 35.8         | 8.7          |
| MobileOne_S0_Checkpoint.IMAGENETTE      | 88.08%    | 98.83%    | 4.3M       | 16.9          | 6.7          | 1.7          |
| MobileOne_S1_Checkpoint.IMAGENETTE      | 91.26%    | 99.18%    | 3.6M       | 13.9          | 11.8         | 1.9          |
| MobileOne_S2_Checkpoint.IMAGENETTE      | 91.31%    | 99.21%    | 5.9M       | 22.8          | 14.4         | 2.4          |
| MobileOne_S3_Checkpoint.IMAGENETTE      | 91.06%    | 99.31%    | 8.1M       | 31.5          | 17.8         | 3.0          |
| ReXNet1_0x_Checkpoint.IMAGENET1K        | 77.86%    | 93.87%    | 4.8M       | 18.5          | 19.8         | 6.5          |
| ReXNet1_0x_Checkpoint.IMAGENETTE        | 94.39%    | 99.62%    | 3.5M       | 13.7          | 19.8         | 6.5          |
| ReXNet1_3x_Checkpoint.IMAGENET1K        | 79.50%    | 94.68%    | 7.6M       | 29.1          | 26.1         | 8.2          |
| ReXNet1_3x_Checkpoint.IMAGENETTE        | 94.88%    | 99.39%    | 5.9M       | 22.8          | 26.1         | 8.2          |
| ReXNet1_5x_Checkpoint.IMAGENET1K        | 80.31%    | 95.17%    | 9.7M       | 37.4          | 30.3         | 6.6          |
| ReXNet1_5x_Checkpoint.IMAGENETTE        | 94.47%    | 99.62%    | 7.8M       | 30.2          | 30.3         | 6.6          |
| ReXNet2_0x_Checkpoint.IMAGENET1K        | 81.64%    | 95.67%    | 16.4M      | 62.8          | 39.5         | 7.7          |
| ReXNet2_0x_Checkpoint.IMAGENETTE        | 95.24%    | 99.57%    | 13.8M      | 53.1          | 39.5         | 7.7          |
| ReXNet2_2x_Checkpoint.IMAGENETTE        | 95.44%    | 99.46%    | 16.7M      | 64.1          | 43.8         | 8.6          |
| RepVGG_A0_Checkpoint.IMAGENETTE         | 92.92%    | 99.46%    | 24.7M      | 94.6          | 17.8         | 3.5          |
| RepVGG_A1_Checkpoint.IMAGENETTE         | 93.78%    | 99.18%    | 30.1M      | 115.1         | 24.2         | 4.6          |
| RepVGG_A2_Checkpoint.IMAGENETTE         | 93.63%    | 99.39%    | 48.6M      | 185.8         | 35.8         | 7.9          |
| RepVGG_B0_Checkpoint.IMAGENETTE         | 92.69%    | 99.21%    | 31.8M      | 121.8         | 28.6         | 5.0          |
| RepVGG_B1_Checkpoint.IMAGENETTE         | 93.96%    | 99.39%    | 100.8M     | 385.1         | 62.4         | 13.6         |
| RepVGG_B2_Checkpoint.IMAGENETTE         | 94.14%    | 99.57%    | 157.5M     | 601.2         | 88.1         | 19.5         |
| Res2Net50_26w_4s_Checkpoint.IMAGENETTE  | 93.94%    | 99.41%    | 23.7M      | 90.6          | 29.1         | 9.2          |
| ResNeXt50_32x4d_Checkpoint.IMAGENETTE   | 94.55%    | 99.49%    | 23.0M      | 88.1          | 33.2         | 7.4          |
| ResNet18_Checkpoint.IMAGENETTE          | 93.61%    | 99.46%    | 11.2M      | 42.7          | 16.4         | 2.7          |
| ResNet34_Checkpoint.IMAGENETTE          | 93.81%    | 99.49%    | 21.3M      | 81.3          | 24.9         | 4.4          |
| ResNet50D_Checkpoint.IMAGENETTE         | 94.65%    | 99.52%    | 23.5M      | 90.1          | 28.4         | 7.6          |
| ResNet50_Checkpoint.IMAGENETTE          | 93.78%    | 99.54%    | 23.5M      | 90            | 26.7         | 7.1          |
| SKNet50_Checkpoint.IMAGENETTE           | 94.37%    | 99.54%    | 35.2M      | 134.7         | 46.9         | 36.1         |

## Object Detection

Experimental

YOLOv4's corrected implementation passes regression tests and a CPU fixed-batch learning check. Full CUDA/VOC training on the repaired implementation remains pending. No pretrained detection checkpoints or paper-level accuracy results are published.

Object detection models expect a 4D image tensor as an input (N x C x H x W) and returns a list of dictionaries. In evaluation mode, each dictionary has three keys: `boxes` (normalized xmin, ymin, xmax, ymax coordinates), `scores` (objectness multiplied by the top class probability for YOLOv1/v2/v4; class probability for YOLO26), and `labels` (class indices). In training mode, pass a list of target dictionaries with normalized `boxes` and integer `labels`; the model returns a loss dictionary.

```python
import holocron.models as models

yolov2 = models.yolov2(num_classes=10)
```

### YOLO training validation

YOLOv4 uses a CSPDarknet53-Mish backbone with an SPP/PAN neck and three detection scales. Its corrected training uses global and additional anchor matches, constant objectness BCE targets, summed classification BCE, and area-weighted differentiable CIoU. Box geometry is computed in FP32 under AMP, and every scale is decoded before combined-confidence filtering and class-aware NMS.

The default `yolov4(num_classes=20)` loads a pretrained **Imagenette backbone**, even without `pretrained=True`. The detector head starts from scratch. The reference script's `--freeze-until backbone` freezes the pretrained weights; `--pretrained` should not be used to request unavailable detection weights.

The [CPU learning diagnostic](https://github.com/frgfm/holocron/blob/main/references/detection/check_yolov4.py) trains the actual neck and head for 500 SGD updates on two synthetic 96-pixel rectangle images, repeated to a batch of eight. It caches the frozen backbone features and disables DropBlock only for this memorization check. Normal evaluation and default NMS give one correctly labeled detection per image, with IoUs **0.8113 / 0.7945**, no false positives, and finite losses and parameter gradients. The pretrained backbone weights remain unchanged, and a trainer checkpoint is saved.

| Validation                                    | Result                                                                                                         |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| YOLOv4 fixed-batch learning                   | Verified; detection error 99.43% with the old initialization restored, versus 0% with corrected initialization |
| YOLOv1/v2 losses and shared inference         | Loss behavior preserved; regression coverage for combined confidence and class-aware NMS                       |
| YOLOv1/v2/v4 ONNX                             | Fixed-shape export and ONNX Runtime CPU parity tested; accuracy not benchmarked                                |
| Full CUDA/VOC training of the repaired YOLOv4 | Pending; synthetic results do not establish validation-set accuracy                                            |

YOLOv3 is not implemented as a detector. Its Darknet-53 classification backbone is available. The [detection guide](https://github.com/frgfm/holocron/blob/main/references/detection/README.md) contains the matched-control results, reproducible commands, learning-rate finder options, metric definitions, and the five-epoch CUDA/VOC acceptance gate.

### YOLO family

[YOLO26 nano](../detection/yolo26/) adds dual-head training and NMS-free inference, with 2.41M parameters in its fused 80-class inference model. Its full published training recipe and pretrained weights are not included.

#### YOLOv1

```python
YOLOv1(layout: list[list[int]], num_classes: int = 20, in_channels: int = 3, stem_channels: int = 64, num_anchors: int = 2, lambda_obj: float = 1, lambda_noobj: float = 0.5, lambda_class: float = 1, lambda_coords: float = 5.0, rpn_nms_thresh: float = 0.7, box_score_thresh: float = 0.05, head_hidden_nodes: int = 512, act_layer: Module | None = None, norm_layer: Callable[[int], Module] | None = None, drop_layer: Callable[..., Module] | None = None, conv_layer: Callable[..., Module] | None = None, backbone_norm_layer: Callable[[int], Module] | None = None)
```

Source code in `holocron/models/detection/yolo.py`

```python
def __init__(
    self,
    layout: list[list[int]],
    num_classes: int = 20,
    in_channels: int = 3,
    stem_channels: int = 64,
    num_anchors: int = 2,
    lambda_obj: float = 1,
    lambda_noobj: float = 0.5,
    lambda_class: float = 1,
    lambda_coords: float = 5.0,
    rpn_nms_thresh: float = 0.7,
    box_score_thresh: float = 0.05,
    head_hidden_nodes: int = 512,  # In the original paper, 4096
    act_layer: nn.Module | None = None,
    norm_layer: Callable[[int], nn.Module] | None = None,
    drop_layer: Callable[..., nn.Module] | None = None,
    conv_layer: Callable[..., nn.Module] | None = None,
    backbone_norm_layer: Callable[[int], nn.Module] | None = None,
) -> None:
    super().__init__(
        num_classes, rpn_nms_thresh, box_score_thresh, lambda_obj, lambda_noobj, lambda_class, lambda_coords
    )

    if act_layer is None:
        act_layer = nn.LeakyReLU(0.1, inplace=True)

    if backbone_norm_layer is None and norm_layer is not None:
        backbone_norm_layer = norm_layer

    self.backbone = DarknetBodyV1(layout, in_channels, stem_channels, act_layer, backbone_norm_layer)

    self.block4 = nn.Sequential(
        *conv_sequence(
            1024,
            1024,
            act_layer,
            norm_layer,
            drop_layer,
            conv_layer,
            kernel_size=3,
            padding=1,
            bias=(norm_layer is None),
        ),
        *conv_sequence(
            1024,
            1024,
            act_layer,
            norm_layer,
            drop_layer,
            conv_layer,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=(norm_layer is None),
        ),
        *conv_sequence(
            1024,
            1024,
            act_layer,
            norm_layer,
            drop_layer,
            conv_layer,
            kernel_size=3,
            padding=1,
            bias=(norm_layer is None),
        ),
        *conv_sequence(
            1024,
            1024,
            act_layer,
            norm_layer,
            drop_layer,
            conv_layer,
            kernel_size=3,
            padding=1,
            bias=(norm_layer is None),
        ),
    )

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(1024 * 7**2, head_hidden_nodes),
        act_layer,
        nn.Dropout(0.5),
        nn.Linear(head_hidden_nodes, 7**2 * (num_anchors * 5 + num_classes)),
    )
    self.num_anchors: int = num_anchors

    init_module(self.block4, "leaky_relu")
    init_module(self.classifier, "leaky_relu")
```

#### yolov1

```python
yolov1(pretrained: bool = False, progress: bool = True, pretrained_backbone: bool = True, **kwargs: Any) -> YOLOv1
```

YOLO model from ["You Only Look Once: Unified, Real-Time Object Detection"](https://pjreddie.com/media/files/papers/yolo_1.pdf).

YOLO's particularity is to make predictions in a grid (same size as last feature map). For each grid cell, the model predicts classification scores and a fixed number of boxes (default: 2). Each box in the cell gets 5 predictions: an objectness score, and 4 coordinates. The 4 coordinates are composed of: the 2-D coordinates of the predicted box center (relative to the cell), and the width and height of the predicted box (relative to the whole image).

For training, YOLO uses a multi-part loss whose components are computed by:

\[ \\mathcal{L}_{coords} = \\sum\\limits_{i=0}^{S^2} \\sum\\limits\_{j=0}^{B} \\mathbb{1}_{ij}^{obj} \\Big\[ (x_{ij} - \\hat{x}_{ij})² + (y_{ij} - \\hat{y}_{ij})² + (\\sqrt{w_{ij}} - \\sqrt{\\hat{w}_{ij}})² + (\\sqrt{h_{ij}} - \\sqrt{\\hat{h}\_{ij}})² \\Big\] \]

where: (S) is size of the output feature map (7 for an input size ((448, 448))), (B) is the number of anchor boxes per grid cell (default: 2), (\\mathbb{1}_{ij}^{obj}) equals to 1 if a GT center falls inside the i-th grid cell and among the anchor boxes of that cell, has the highest IoU with the j-th box else 0, ((x_{ij}, y\_{ij}, w\_{ij}, h\_{ij})) are the coordinates of the ground truth assigned to the j-th anchor box of the i-th grid cell, ((\\hat{x}_{ij}, \\hat{y}_{ij}, \\hat{w}_{ij}, \\hat{h}_{ij})) are the coordinate predictions for the j-th anchor box of the i-th grid cell.

\[ \\mathcal{L}_{objectness} = \\sum\\limits_{i=0}^{S^2} \\sum\\limits\_{j=0}^{B} \\Big\[ \\mathbb{1}_{ij}^{obj} \\Big(C_{ij} - \\hat{C}\_{ij} \\Big)^2

- \\lambda\_{noobj} \\mathbb{1}_{ij}^{noobj} \\Big(C_{ij} - \\hat{C}\_{ij} \\Big)^2 \\Big\] \]

where (\\lambda\_{noobj}) is a positive coefficient (default: 0.5), (\\mathbb{1}_{ij}^{noobj} = 1 - \\mathbb{1}_{ij}^{obj}), (C\_{ij}) equals the Intersection Over Union between the j-th anchor box in the i-th grid cell and its matched ground truth box if that box is matched with a ground truth else 0, and (\\hat{C}\_{ij}) is the objectness score of the j-th anchor box in the i-th grid cell..

\[ \\mathcal{L}_{classification} = \\sum\\limits_{i=0}^{S^2} \\mathbb{1}_{i}^{obj} \\sum\\limits_{c \\in classes} (p_i(c) - \\hat{p}\_i(c))^2 \]

where (\\mathbb{1}\_{i}^{obj}) equals to 1 if a GT center falls inside the i-th grid cell else 0, (p_i(c)) equals 1 if the assigned ground truth to the i-th cell is classified as class (c), and (\\hat{p}\_i(c)) is the predicted probability of class (c) in the i-th cell.

And the full loss is given by:

\[ \\mathcal{L}_{YOLOv1} = \\lambda_{coords} \\cdot \\mathcal{L}_{coords} + \\mathcal{L}_{objectness} + \\mathcal{L}\_{classification} \]

where (\\lambda\_{coords}) is a positive coefficient (default: 5).

| PARAMETER             | DESCRIPTION                                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------------------- |
| `pretrained`          | If True, returns a model pre-trained on ImageNet **TYPE:** `bool` **DEFAULT:** `False`                    |
| `progress`            | If True, displays a progress bar of the download to stderr **TYPE:** `bool` **DEFAULT:** `True`           |
| `pretrained_backbone` | If True, backbone parameters will have been pretrained on Imagenette **TYPE:** `bool` **DEFAULT:** `True` |
| `kwargs`              | keyword args of YOLOv1 **TYPE:** `Any` **DEFAULT:** `{}`                                                  |

| RETURNS  | DESCRIPTION      |
| -------- | ---------------- |
| `YOLOv1` | detection module |

Source code in `holocron/models/detection/yolo.py`

```python
def yolov1(pretrained: bool = False, progress: bool = True, pretrained_backbone: bool = True, **kwargs: Any) -> YOLOv1:
    r"""YOLO model from
    ["You Only Look Once: Unified, Real-Time Object Detection"](https://pjreddie.com/media/files/papers/yolo_1.pdf).

    YOLO's particularity is to make predictions in a grid (same size as last feature map). For each grid cell,
    the model predicts classification scores and a fixed number of boxes (default: 2). Each box in the cell gets
    5 predictions: an objectness score, and 4 coordinates. The 4 coordinates are composed of: the 2-D coordinates of
    the predicted box center (relative to the cell), and the width and height of the predicted box (relative to
    the whole image).

    For training, YOLO uses a multi-part loss whose components are computed by:

    $$
    \mathcal{L}_{coords} = \sum\limits_{i=0}^{S^2} \sum\limits_{j=0}^{B}
    \mathbb{1}_{ij}^{obj} \Big[
    (x_{ij} - \hat{x}_{ij})² + (y_{ij} - \hat{y}_{ij})² +
    (\sqrt{w_{ij}} - \sqrt{\hat{w}_{ij}})² + (\sqrt{h_{ij}} - \sqrt{\hat{h}_{ij}})²
    \Big]
    $$

    where:
    $S$ is size of the output feature map (7 for an input size $(448, 448)$),
    $B$ is the number of anchor boxes per grid cell (default: 2),
    $\mathbb{1}_{ij}^{obj}$ equals to 1 if a GT center falls inside the i-th grid cell and among the
    anchor boxes of that cell, has the highest IoU with the j-th box else 0,
    $(x_{ij}, y_{ij}, w_{ij}, h_{ij})$ are the coordinates of the ground truth assigned to
    the j-th anchor box of the i-th grid cell,
    $(\hat{x}_{ij}, \hat{y}_{ij}, \hat{w}_{ij}, \hat{h}_{ij})$ are the coordinate predictions
    for the j-th anchor box of the i-th grid cell.

    $$
    \mathcal{L}_{objectness} = \sum\limits_{i=0}^{S^2} \sum\limits_{j=0}^{B}
    \Big[ \mathbb{1}_{ij}^{obj} \Big(C_{ij} - \hat{C}_{ij} \Big)^2
    + \lambda_{noobj} \mathbb{1}_{ij}^{noobj} \Big(C_{ij} - \hat{C}_{ij} \Big)^2
    \Big]
    $$

    where $\lambda_{noobj}$ is a positive coefficient (default: 0.5),
    $\mathbb{1}_{ij}^{noobj} = 1 - \mathbb{1}_{ij}^{obj}$,
    $C_{ij}$ equals the Intersection Over Union between the j-th anchor box in the i-th grid cell and its
    matched ground truth box if that box is matched with a ground truth else 0,
    and $\hat{C}_{ij}$ is the objectness score of the j-th anchor box in the i-th grid cell..

    $$
    \mathcal{L}_{classification} = \sum\limits_{i=0}^{S^2}
    \mathbb{1}_{i}^{obj} \sum\limits_{c \in classes}
    (p_i(c) - \hat{p}_i(c))^2
    $$

    where $\mathbb{1}_{i}^{obj}$ equals to 1 if a GT center falls inside the i-th grid cell else 0,
    $p_i(c)$ equals 1 if the assigned ground truth to the i-th cell is classified as class $c$,
    and $\hat{p}_i(c)$ is the predicted probability of class $c$ in the i-th cell.

    And the full loss is given by:

    $$
    \mathcal{L}_{YOLOv1} = \lambda_{coords} \cdot \mathcal{L}_{coords} +
    \mathcal{L}_{objectness} + \mathcal{L}_{classification}
    $$

    where $\lambda_{coords}$ is a positive coefficient (default: 5).

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        progress: If True, displays a progress bar of the download to stderr
        pretrained_backbone: If True, backbone parameters will have been pretrained on Imagenette
        kwargs: keyword args of [`YOLOv1`][holocron.models.detection.yolo.YOLOv1]

    Returns:
        detection module
    """
    return _yolo(
        "yolov1",
        pretrained,
        progress,
        pretrained_backbone,
        [[192], [128, 256, 256, 512], [*([256, 512] * 4), 512, 1024], [512, 1024] * 2],
        **kwargs,
    )
```

#### YOLOv2

```python
YOLOv2(layout: list[tuple[int, int]], num_classes: int = 20, in_channels: int = 3, stem_chanels: int = 32, anchors: Tensor | None = None, passthrough_ratio: int = 8, lambda_obj: float = 1, lambda_noobj: float = 0.5, lambda_class: float = 1, lambda_coords: float = 5, rpn_nms_thresh: float = 0.7, box_score_thresh: float = 0.05, act_layer: Module | None = None, norm_layer: Callable[[int], Module] | None = None, drop_layer: Callable[..., Module] | None = None, conv_layer: Callable[..., Module] | None = None, backbone_norm_layer: Callable[[int], Module] | None = None)
```

Source code in `holocron/models/detection/yolov2.py`

```python
def __init__(
    self,
    layout: list[tuple[int, int]],
    num_classes: int = 20,
    in_channels: int = 3,
    stem_chanels: int = 32,
    anchors: Tensor | None = None,
    passthrough_ratio: int = 8,
    lambda_obj: float = 1,
    lambda_noobj: float = 0.5,
    lambda_class: float = 1,
    lambda_coords: float = 5,
    rpn_nms_thresh: float = 0.7,
    box_score_thresh: float = 0.05,
    act_layer: nn.Module | None = None,
    norm_layer: Callable[[int], nn.Module] | None = None,
    drop_layer: Callable[..., nn.Module] | None = None,
    conv_layer: Callable[..., nn.Module] | None = None,
    backbone_norm_layer: Callable[[int], nn.Module] | None = None,
) -> None:
    super().__init__(
        num_classes, rpn_nms_thresh, box_score_thresh, lambda_obj, lambda_noobj, lambda_class, lambda_coords
    )

    if act_layer is None:
        act_layer = nn.LeakyReLU(0.1, inplace=True)
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    if backbone_norm_layer is None:
        backbone_norm_layer = norm_layer

    # Priors computed using K-means
    if anchors is None:
        # cf. https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-voc.cfg#L242
        anchors = (
            torch.tensor([
                [1.3221, 1.73145],
                [3.19275, 4.00944],
                [5.05587, 8.09892],
                [9.47112, 4.84053],
                [11.2364, 10.0071],
            ])
            / 13
        )

    self.backbone = DarknetBodyV2(
        layout, in_channels, stem_chanels, True, act_layer, backbone_norm_layer, drop_layer, conv_layer
    )

    self.block5 = nn.Sequential(
        *conv_sequence(
            layout[-1][0],
            layout[-1][0],
            act_layer,
            norm_layer,
            drop_layer,
            conv_layer,
            kernel_size=3,
            padding=1,
            bias=(norm_layer is None),
        ),
        *conv_sequence(
            layout[-1][0],
            layout[-1][0],
            act_layer,
            norm_layer,
            drop_layer,
            conv_layer,
            kernel_size=3,
            padding=1,
            bias=(norm_layer is None),
        ),
    )

    self.passthrough_layer = nn.Sequential(
        *conv_sequence(
            layout[-2][0],
            layout[-2][0] // passthrough_ratio,
            act_layer,
            norm_layer,
            drop_layer,
            conv_layer,
            kernel_size=1,
            bias=(norm_layer is None),
        ),
        ConcatDownsample2d(scale_factor=2),
    )

    self.block6 = nn.Sequential(
        *conv_sequence(
            layout[-1][0] + layout[-2][0] // passthrough_ratio * 2**2,
            layout[-1][0],
            act_layer,
            norm_layer,
            drop_layer,
            conv_layer,
            kernel_size=3,
            padding=1,
            bias=(norm_layer is None),
        )
    )

    # Each box has P_objectness, 4 coords, and score for each class
    self.head = nn.Conv2d(layout[-1][0], anchors.shape[0] * (5 + num_classes), 1)

    # Register losses
    self.register_buffer("anchors", anchors)

    init_module(self.block5, "leaky_relu")
    init_module(self.passthrough_layer, "leaky_relu")
    init_module(self.block6, "leaky_relu")
    # Initialize the head like a linear (default Conv2D init is the same as Linear)
    if self.head.bias is not None:
        self.head.bias.data.zero_()
```

#### yolov2

```python
yolov2(pretrained: bool = False, progress: bool = True, pretrained_backbone: bool = True, **kwargs: Any) -> YOLOv2
```

YOLOv2 model from ["YOLO9000: Better, Faster, Stronger"](https://pjreddie.com/media/files/papers/YOLO9000.pdf).

YOLOv2 improves upon YOLO by raising the number of boxes predicted by grid cell (default: 5), introducing bounding box priors and predicting class scores for each anchor box in the grid cell.

For training, YOLOv2 uses the same multi-part loss as YOLO apart from its classification loss:

\[ \\mathcal{L}_{classification} = \\sum\\limits_{i=0}^{S^2} \\sum\\limits\_{j=0}^{B} \\mathbb{1}_{ij}^{obj} \\sum\\limits_{c \\in classes} (p\_{ij}(c) - \\hat{p}\_{ij}(c))^2 \]

where:

- (S) is size of the output feature map (13 for an input size ((416, 416))),
- (B) is the number of anchor boxes per grid cell (default: 5),
- (\\mathbb{1}\_{ij}^{obj}) equals to 1 if a GT center falls inside the i-th grid cell and among the anchor boxes of that cell, has the highest IoU with the j-th box else 0,
- (p\_{ij}(c)) equals 1 if the assigned ground truth to the j-th anchor box of the i-th cell is classified as class (c),
- (\\hat{p}\_{ij}(c)) is the predicted probability of class (c) for the j-th anchor box in the i-th cell.

| PARAMETER             | DESCRIPTION                                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------------------- |
| `pretrained`          | If True, returns a model pre-trained on ImageNet **TYPE:** `bool` **DEFAULT:** `False`                    |
| `progress`            | If True, displays a progress bar of the download to stderr **TYPE:** `bool` **DEFAULT:** `True`           |
| `pretrained_backbone` | If True, backbone parameters will have been pretrained on Imagenette **TYPE:** `bool` **DEFAULT:** `True` |
| `kwargs`              | keyword args of YOLOv2 **TYPE:** `Any` **DEFAULT:** `{}`                                                  |

| RETURNS  | DESCRIPTION      |
| -------- | ---------------- |
| `YOLOv2` | detection module |

Source code in `holocron/models/detection/yolov2.py`

```python
def yolov2(pretrained: bool = False, progress: bool = True, pretrained_backbone: bool = True, **kwargs: Any) -> YOLOv2:
    r"""YOLOv2 model from
    ["YOLO9000: Better, Faster, Stronger"](https://pjreddie.com/media/files/papers/YOLO9000.pdf).

    YOLOv2 improves upon YOLO by raising the number of boxes predicted by grid cell (default: 5), introducing
    bounding box priors and predicting class scores for each anchor box in the grid cell.

    For training, YOLOv2 uses the same multi-part loss as YOLO apart from its classification loss:

    $$
    \mathcal{L}_{classification} = \sum\limits_{i=0}^{S^2}  \sum\limits_{j=0}^{B}
    \mathbb{1}_{ij}^{obj} \sum\limits_{c \in classes}
    (p_{ij}(c) - \hat{p}_{ij}(c))^2
    $$

    where:
    - $S$ is size of the output feature map (13 for an input size $(416, 416)$),
    - $B$ is the number of anchor boxes per grid cell (default: 5),
    - $\mathbb{1}_{ij}^{obj}$ equals to 1 if a GT center falls inside the i-th grid cell and among the
    anchor boxes of that cell, has the highest IoU with the j-th box else 0,
    - $p_{ij}(c)$ equals 1 if the assigned ground truth to the j-th anchor box of the i-th cell is classified
    as class $c$,
    - $\hat{p}_{ij}(c)$ is the predicted probability of class $c$ for the j-th anchor box
    in the i-th cell.

    Args:
        pretrained: If True, returns a model pre-trained on ImageNet
        progress: If True, displays a progress bar of the download to stderr
        pretrained_backbone: If True, backbone parameters will have been pretrained on Imagenette
        kwargs: keyword args of [`YOLOv2`][holocron.models.detection.yolov2.YOLOv2]

    Returns:
        detection module
    """
    if pretrained_backbone:
        kwargs["backbone_norm_layer"] = FrozenBatchNorm2d

    return _yolo(
        "yolov2",
        pretrained,
        progress,
        pretrained_backbone,
        [(64, 0), (128, 1), (256, 1), (512, 2), (1024, 2)],
        **kwargs,
    )
```

#### YOLOv4

```python
YOLOv4(layout: list[tuple[int, int]], num_classes: int = 80, in_channels: int = 3, stem_channels: int = 32, anchors: Tensor | None = None, act_layer: Module | None = None, norm_layer: Callable[[int], Module] | None = None, drop_layer: Callable[..., Module] | None = None, conv_layer: Callable[..., Module] | None = None, backbone_norm_layer: Callable[[int], Module] | None = None)
```

Source code in `holocron/models/detection/yolov4.py`

```python
def __init__(
    self,
    layout: list[tuple[int, int]],
    num_classes: int = 80,
    in_channels: int = 3,
    stem_channels: int = 32,
    anchors: Tensor | None = None,
    act_layer: nn.Module | None = None,
    norm_layer: Callable[[int], nn.Module] | None = None,
    drop_layer: Callable[..., nn.Module] | None = None,
    conv_layer: Callable[..., nn.Module] | None = None,
    backbone_norm_layer: Callable[[int], nn.Module] | None = None,
) -> None:
    super().__init__()

    if act_layer is None:
        act_layer = nn.Mish(inplace=True)
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    if backbone_norm_layer is None:
        backbone_norm_layer = norm_layer
    if drop_layer is None:
        drop_layer = DropBlock2d

    # backbone
    self.backbone = DarknetBodyV4(
        layout, in_channels, stem_channels, 3, act_layer, backbone_norm_layer, drop_layer, conv_layer
    )
    # neck
    self.neck = Neck([1024, 512, 256], act_layer, norm_layer, drop_layer, conv_layer)
    # head
    self.head = Yolov4Head(num_classes, anchors, act_layer, norm_layer, drop_layer, conv_layer)
```

#### yolov4

```python
yolov4(pretrained: bool = False, progress: bool = True, pretrained_backbone: bool = True, **kwargs: Any) -> YOLOv4
```

YOLOv4 model from ["YOLOv4: Optimal Speed and Accuracy of Object Detection"](https://arxiv.org/pdf/2004.10934.pdf).

The implementation combines a CSPDarknet53-Mish backbone with an SPP/PAN neck and three detection scales. Training uses the YOLOv4 anchor masks and multi-anchor assignment, binary cross-entropy for objectness and classification, and area-weighted Complete IoU for matched boxes.

| PARAMETER             | DESCRIPTION                                                                                                       |
| --------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `pretrained`          | Whether to request pretrained detector parameters (not currently available) **TYPE:** `bool` **DEFAULT:** `False` |
| `progress`            | If True, displays a progress bar of the download to stderr **TYPE:** `bool` **DEFAULT:** `True`                   |
| `pretrained_backbone` | If True, backbone parameters will have been pretrained on Imagenette **TYPE:** `bool` **DEFAULT:** `True`         |
| `kwargs`              | keyword args of YOLOv4 **TYPE:** `Any` **DEFAULT:** `{}`                                                          |

| RETURNS  | DESCRIPTION      |
| -------- | ---------------- |
| `YOLOv4` | detection module |

Source code in `holocron/models/detection/yolov4.py`

```python
def yolov4(pretrained: bool = False, progress: bool = True, pretrained_backbone: bool = True, **kwargs: Any) -> YOLOv4:
    r"""YOLOv4 model from
    ["YOLOv4: Optimal Speed and Accuracy of Object Detection"](https://arxiv.org/pdf/2004.10934.pdf).

    The implementation combines a CSPDarknet53-Mish backbone with an SPP/PAN neck and three detection scales.
    Training uses the YOLOv4 anchor masks and multi-anchor assignment, binary cross-entropy for objectness and
    classification, and area-weighted Complete IoU for matched boxes.

    Args:
        pretrained: Whether to request pretrained detector parameters (not currently available)
        progress: If True, displays a progress bar of the download to stderr
        pretrained_backbone: If True, backbone parameters will have been pretrained on Imagenette
        kwargs: keyword args of [`YOLOv4`][holocron.models.detection.yolov4.YOLOv4]

    Returns:
        detection module
    """
    if pretrained_backbone:
        kwargs["backbone_norm_layer"] = FrozenBatchNorm2d

    return _yolo(
        "yolov4",
        pretrained,
        progress,
        pretrained_backbone,
        [(64, 1), (128, 2), (256, 8), (512, 8), (1024, 4)],
        **kwargs,
    )
```

## Semantic Segmentation

Unbenchmarked

`unet_rexnet13` is the only segmentation model with a legacy checkpoint. Its training dataset and evaluation metric are not documented. Other segmentation architectures have no published weights.

Semantic segmentation models expect a 4D image tensor as an input (N x C x H x W) and returns a classification score tensor of size (N x K x Ho x Wo).

```python
import holocron.models as models

unet = models.unet(num_classes=10)
```

### U-Net family

#### UNet

```python
UNet(layout: list[int], in_channels: int = 3, num_classes: int = 10, act_layer: Module | None = None, norm_layer: Callable[[int], Module] | None = None, drop_layer: Callable[..., Module] | None = None, conv_layer: Callable[..., Module] | None = None, same_padding: bool = True, bilinear_upsampling: bool = True)
```

Implements a U-Net architecture

| PARAMETER             | DESCRIPTION                                                                                            |
| --------------------- | ------------------------------------------------------------------------------------------------------ |
| `layout`              | number of channels after each contracting block **TYPE:** `list[int]`                                  |
| `in_channels`         | number of channels in the input tensor **TYPE:** `int` **DEFAULT:** `3`                                |
| `num_classes`         | number of output classes **TYPE:** `int` **DEFAULT:** `10`                                             |
| `act_layer`           | activation layer **TYPE:** \`Module                                                                    |
| `norm_layer`          | normalization layer **TYPE:** \`Callable\[[int], Module\]                                              |
| `drop_layer`          | dropout layer **TYPE:** \`Callable[..., Module]                                                        |
| `conv_layer`          | convolutional layer **TYPE:** \`Callable[..., Module]                                                  |
| `same_padding`        | enforces same padding in convolutions **TYPE:** `bool` **DEFAULT:** `True`                             |
| `bilinear_upsampling` | replaces transposed conv by bilinear interpolation for upsampling **TYPE:** `bool` **DEFAULT:** `True` |

Source code in `holocron/models/segmentation/unet.py`

```python
def __init__(
    self,
    layout: list[int],
    in_channels: int = 3,
    num_classes: int = 10,
    act_layer: nn.Module | None = None,
    norm_layer: Callable[[int], nn.Module] | None = None,
    drop_layer: Callable[..., nn.Module] | None = None,
    conv_layer: Callable[..., nn.Module] | None = None,
    same_padding: bool = True,
    bilinear_upsampling: bool = True,
) -> None:
    super().__init__()

    if act_layer is None:
        act_layer = nn.ReLU(inplace=True)

    # Contracting path
    self.encoder = nn.ModuleList([])
    layout_ = [in_channels, *layout]
    pool = False
    for in_chan, out_chan in pairwise(layout_):
        self.encoder.append(
            down_path(in_chan, out_chan, pool, int(same_padding), act_layer, norm_layer, drop_layer, conv_layer)
        )
        pool = True

    self.bridge = nn.Sequential(
        nn.MaxPool2d((2, 2)),
        *conv_sequence(
            layout[-1],
            2 * layout[-1],
            act_layer,
            norm_layer,
            drop_layer,
            conv_layer,
            kernel_size=3,
            padding=int(same_padding),
        ),
        *conv_sequence(
            2 * layout[-1],
            layout[-1],
            act_layer,
            norm_layer,
            drop_layer,
            conv_layer,
            kernel_size=3,
            padding=int(same_padding),
        ),
    )

    # Expansive path
    self.decoder = nn.ModuleList([])
    layout_ = [chan // 2 if bilinear_upsampling else chan for chan in layout[::-1][:-1]] + [layout[0]]
    up_chan = layout[-1]
    for left_chan, out_chan in zip(layout[::-1], layout_, strict=True):
        self.decoder.append(
            UpPath(
                left_chan + up_chan,
                out_chan,
                bilinear_upsampling,
                int(same_padding),
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
                up_chan=up_chan,
            )
        )
        up_chan = out_chan

    # Classifier
    self.classifier = nn.Conv2d(layout[0], num_classes, 1)

    init_module(self, "relu")
```

#### unet

```python
unet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNet
```

U-Net from ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf)

| PARAMETER    | DESCRIPTION                                                                                     |
| ------------ | ----------------------------------------------------------------------------------------------- |
| `pretrained` | If True, returns a model pre-trained on PASCAL VOC2012 **TYPE:** `bool` **DEFAULT:** `False`    |
| `progress`   | If True, displays a progress bar of the download to stderr **TYPE:** `bool` **DEFAULT:** `True` |
| `kwargs`     | keyword args of UNet **TYPE:** `Any` **DEFAULT:** `{}`                                          |

| RETURNS | DESCRIPTION                 |
| ------- | --------------------------- |
| `UNet`  | semantic segmentation model |

Source code in `holocron/models/segmentation/unet.py`

```python
def unet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNet:
    """U-Net from
    ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf)

    ![UNet explanation](https://github.com/frgfm/Holocron/releases/download/v0.1.3/unet.png)

    Args:
        pretrained: If True, returns a model pre-trained on PASCAL VOC2012
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of [`UNet`][holocron.models.segmentation.unet.UNet]

    Returns:
        semantic segmentation model
    """
    return _unet("unet", pretrained, progress, **kwargs)
```

#### DynamicUNet

```python
DynamicUNet(encoder: IntermediateLayerGetter, num_classes: int = 10, act_layer: Module | None = None, norm_layer: Callable[[int], Module] | None = None, drop_layer: Callable[..., Module] | None = None, conv_layer: Callable[..., Module] | None = None, same_padding: bool = True, input_shape: tuple[int, int, int] | None = None, final_upsampling: bool = False)
```

Implements a dynamic U-Net architecture

| PARAMETER          | DESCRIPTION                                                                                              |
| ------------------ | -------------------------------------------------------------------------------------------------------- |
| `encoder`          | feature extractor used for encoding **TYPE:** `IntermediateLayerGetter`                                  |
| `num_classes`      | number of output classes **TYPE:** `int` **DEFAULT:** `10`                                               |
| `act_layer`        | activation layer **TYPE:** \`Module                                                                      |
| `norm_layer`       | normalization layer **TYPE:** \`Callable\[[int], Module\]                                                |
| `drop_layer`       | dropout layer **TYPE:** \`Callable[..., Module]                                                          |
| `conv_layer`       | convolutional layer **TYPE:** \`Callable[..., Module]                                                    |
| `same_padding`     | enforces same padding in convolutions **TYPE:** `bool` **DEFAULT:** `True`                               |
| `input_shape`      | shape of the input tensor **TYPE:** \`tuple[int, int, int]                                               |
| `final_upsampling` | whether to upsample final decoder features to the input resolution **TYPE:** `bool` **DEFAULT:** `False` |

Source code in `holocron/models/segmentation/unet.py`

```python
def __init__(
    self,
    encoder: IntermediateLayerGetter,
    num_classes: int = 10,
    act_layer: nn.Module | None = None,
    norm_layer: Callable[[int], nn.Module] | None = None,
    drop_layer: Callable[..., nn.Module] | None = None,
    conv_layer: Callable[..., nn.Module] | None = None,
    same_padding: bool = True,
    input_shape: tuple[int, int, int] | None = None,
    final_upsampling: bool = False,
) -> None:
    super().__init__()

    if act_layer is None:
        act_layer = nn.ReLU(inplace=True)

    self.encoder = encoder
    # Determine all feature map shapes
    training_mode = self.encoder.training
    self.encoder.eval()
    input_shape = (3, 256, 256) if input_shape is None else input_shape
    with torch.no_grad():
        shapes = [v.shape[1:] for v in self.encoder(torch.zeros(1, *input_shape)).values()]
    chans = [s[0] for s in shapes]
    if training_mode:
        self.encoder.train()

    # Middle layers
    self.bridge = nn.Sequential(
        nn.BatchNorm2d(chans[-1]) if norm_layer is None else norm_layer(chans[-1]),
        act_layer,
        *conv_sequence(
            chans[-1], 2 * chans[-1], act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3, padding=1
        ),
        *conv_sequence(
            2 * chans[-1], chans[-1], act_layer, norm_layer, drop_layer, conv_layer, kernel_size=3, padding=1
        ),
    )

    # Expansive path
    self.decoder = nn.ModuleList([])
    layout = [*chans[::-1][1:], chans[0]]
    for up_chan, out_chan in zip(chans[::-1], layout, strict=True):
        self.decoder.append(
            UBlock(up_chan, up_chan, out_chan, int(same_padding), act_layer, norm_layer, drop_layer, conv_layer)
        )

    # Final upsampling if sizes don't match
    self.upsample: nn.Sequential | None = None
    if final_upsampling:
        self.upsample = nn.Sequential(
            *conv_sequence(chans[0], chans[0] * 2**2, act_layer, norm_layer, drop_layer, conv_layer, kernel_size=1),
            nn.PixelShuffle(upscale_factor=2),
        )

    # Classifier
    self.classifier = nn.Conv2d(chans[0], num_classes, 1)

    for module in (self.bridge, self.decoder, self.upsample, self.classifier):
        if module is not None:
            init_module(module, "relu")
```

#### unet2

```python
unet2(pretrained: bool = False, progress: bool = True, in_channels: int = 3, **kwargs: Any) -> DynamicUNet
```

Modified version of U-Net from ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf) that includes a more advanced upscaling block inspired by [fastai](https://docs.fast.ai/vision.models.unet.html#DynamicUnet)

| PARAMETER     | DESCRIPTION                                                                                     |
| ------------- | ----------------------------------------------------------------------------------------------- |
| `pretrained`  | If True, returns a model pre-trained on PASCAL VOC2012 **TYPE:** `bool` **DEFAULT:** `False`    |
| `progress`    | If True, displays a progress bar of the download to stderr **TYPE:** `bool` **DEFAULT:** `True` |
| `in_channels` | number of input channels **TYPE:** `int` **DEFAULT:** `3`                                       |
| `kwargs`      | keyword args of DynamicUNet **TYPE:** `Any` **DEFAULT:** `{}`                                   |

| RETURNS       | DESCRIPTION                 |
| ------------- | --------------------------- |
| `DynamicUNet` | semantic segmentation model |

Source code in `holocron/models/segmentation/unet.py`

```python
def unet2(pretrained: bool = False, progress: bool = True, in_channels: int = 3, **kwargs: Any) -> DynamicUNet:
    """Modified version of U-Net from
    ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf)
    that includes a more advanced upscaling block inspired by [fastai](https://docs.fast.ai/vision.models.unet.html#DynamicUnet)

    ![UNet architecture](https://github.com/frgfm/Holocron/releases/download/v0.1.3/unet.png)

    Args:
        pretrained: If True, returns a model pre-trained on PASCAL VOC2012
        progress: If True, displays a progress bar of the download to stderr
        in_channels: number of input channels
        kwargs: keyword args of [`DynamicUNet`][holocron.models.segmentation.unet.DynamicUNet]

    Returns:
        semantic segmentation model
    """
    backbone = UNetBackbone(default_cfgs["unet2"]["encoder_layout"], in_channels=in_channels).features
    kwargs["input_shape"] = kwargs.get("input_shape") or (in_channels, 256, 256)

    return _dynamic_unet("unet2", backbone, pretrained, progress, **kwargs)  # ty: ignore[invalid-argument-type]
```

#### unet_tvvgg11

```python
unet_tvvgg11(pretrained: bool = False, pretrained_backbone: bool = True, progress: bool = True, **kwargs: Any) -> DynamicUNet
```

U-Net from ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf) with a VGG-11 backbone used as encoder, and more advanced upscaling blocks inspired by [fastai](https://docs.fast.ai/vision.models.unet.html#DynamicUnet)

| PARAMETER             | DESCRIPTION                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------------- |
| `pretrained`          | If True, returns a model pre-trained on PASCAL VOC2012 **TYPE:** `bool` **DEFAULT:** `False`            |
| `pretrained_backbone` | If True, the encoder will load pretrained parameters from ImageNet **TYPE:** `bool` **DEFAULT:** `True` |
| `progress`            | If True, displays a progress bar of the download to stderr **TYPE:** `bool` **DEFAULT:** `True`         |
| `kwargs`              | keyword args of DynamicUNet **TYPE:** `Any` **DEFAULT:** `{}`                                           |

| RETURNS       | DESCRIPTION                 |
| ------------- | --------------------------- |
| `DynamicUNet` | semantic segmentation model |

Source code in `holocron/models/segmentation/unet.py`

```python
def unet_tvvgg11(
    pretrained: bool = False, pretrained_backbone: bool = True, progress: bool = True, **kwargs: Any
) -> DynamicUNet:
    """U-Net from
    ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf)
    with a VGG-11 backbone used as encoder, and more advanced upscaling blocks inspired by [fastai](https://docs.fast.ai/vision.models.unet.html#DynamicUnet)

    Args:
        pretrained: If True, returns a model pre-trained on PASCAL VOC2012
        pretrained_backbone: If True, the encoder will load pretrained parameters from ImageNet
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of [`DynamicUNet`][holocron.models.segmentation.unet.DynamicUNet]

    Returns:
        semantic segmentation model
    """
    weights = get_model_weights("vgg11").DEFAULT if pretrained_backbone and not pretrained else None  # ty: ignore[unresolved-attribute]
    backbone: nn.Module = get_model("vgg11", weights=weights).features  # ty: ignore[invalid-assignment]

    return _dynamic_unet("unet_vgg11", backbone, pretrained, progress, **kwargs)
```

#### unet_tvresnet34

```python
unet_tvresnet34(pretrained: bool = False, pretrained_backbone: bool = True, progress: bool = True, **kwargs: Any) -> DynamicUNet
```

U-Net from ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf) with a ResNet-34 backbone used as encoder, and more advanced upscaling blocks inspired by [fastai](https://docs.fast.ai/vision.models.unet.html#DynamicUnet)

| PARAMETER             | DESCRIPTION                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------------- |
| `pretrained`          | If True, returns a model pre-trained on PASCAL VOC2012 **TYPE:** `bool` **DEFAULT:** `False`            |
| `pretrained_backbone` | If True, the encoder will load pretrained parameters from ImageNet **TYPE:** `bool` **DEFAULT:** `True` |
| `progress`            | If True, displays a progress bar of the download to stderr **TYPE:** `bool` **DEFAULT:** `True`         |
| `kwargs`              | keyword args of DynamicUNet **TYPE:** `Any` **DEFAULT:** `{}`                                           |

| RETURNS       | DESCRIPTION                 |
| ------------- | --------------------------- |
| `DynamicUNet` | semantic segmentation model |

Source code in `holocron/models/segmentation/unet.py`

```python
def unet_tvresnet34(
    pretrained: bool = False, pretrained_backbone: bool = True, progress: bool = True, **kwargs: Any
) -> DynamicUNet:
    """U-Net from
    ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf)
    with a ResNet-34 backbone used as encoder, and more advanced upscaling blocks inspired by [fastai](https://docs.fast.ai/vision.models.unet.html#DynamicUnet)

    Args:
        pretrained: If True, returns a model pre-trained on PASCAL VOC2012
        pretrained_backbone: If True, the encoder will load pretrained parameters from ImageNet
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of [`DynamicUNet`][holocron.models.segmentation.unet.DynamicUNet]

    Returns:
        semantic segmentation model
    """
    weights = get_model_weights("resnet34").DEFAULT if pretrained_backbone and not pretrained else None  # ty: ignore[unresolved-attribute]
    backbone = get_model("resnet34", weights=weights)
    kwargs["final_upsampling"] = kwargs.get("final_upsampling", True)

    return _dynamic_unet("unet_tvresnet34", backbone, pretrained, progress, **kwargs)
```

#### unet_rexnet13

```python
unet_rexnet13(pretrained: bool = False, pretrained_backbone: bool = True, progress: bool = True, in_channels: int = 3, **kwargs: Any) -> DynamicUNet
```

U-Net from ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf) with a ReXNet-1.3x backbone used as encoder, and more advanced upscaling blocks inspired by [fastai](https://docs.fast.ai/vision.models.unet.html#DynamicUnet).

| PARAMETER             | DESCRIPTION                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------------- |
| `pretrained`          | If True, returns a model pre-trained on PASCAL VOC2012 **TYPE:** `bool` **DEFAULT:** `False`            |
| `pretrained_backbone` | If True, the encoder will load pretrained parameters from ImageNet **TYPE:** `bool` **DEFAULT:** `True` |
| `progress`            | If True, displays a progress bar of the download to stderr **TYPE:** `bool` **DEFAULT:** `True`         |
| `in_channels`         | the number of input channels **TYPE:** `int` **DEFAULT:** `3`                                           |
| `kwargs`              | keyword args of DynamicUNet **TYPE:** `Any` **DEFAULT:** `{}`                                           |

| RETURNS       | DESCRIPTION                 |
| ------------- | --------------------------- |
| `DynamicUNet` | semantic segmentation model |

Source code in `holocron/models/segmentation/unet.py`

```python
def unet_rexnet13(
    pretrained: bool = False,
    pretrained_backbone: bool = True,
    progress: bool = True,
    in_channels: int = 3,
    **kwargs: Any,
) -> DynamicUNet:
    """U-Net from
    ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf)
    with a ReXNet-1.3x backbone used as encoder, and more advanced upscaling blocks inspired by [fastai](https://docs.fast.ai/vision.models.unet.html#DynamicUnet).

    Args:
        pretrained: If True, returns a model pre-trained on PASCAL VOC2012
        pretrained_backbone: If True, the encoder will load pretrained parameters from ImageNet
        progress: If True, displays a progress bar of the download to stderr
        in_channels: the number of input channels
        kwargs: keyword args of [`DynamicUNet`][holocron.models.segmentation.unet.DynamicUNet]

    Returns:
        semantic segmentation model
    """
    backbone = rexnet1_3x(pretrained=pretrained_backbone and not pretrained, in_channels=in_channels).features
    kwargs["input_shape"] = kwargs.get("input_shape") or (in_channels, 256, 256)
    kwargs["final_upsampling"] = kwargs.get("final_upsampling", True)
    kwargs["act_layer"] = kwargs.get("act_layer", nn.SiLU(inplace=True))
    # hotfix of https://github.com/pytorch/vision/issues/3802
    backbone[21] = nn.SiLU(inplace=True)

    return _dynamic_unet("unet_rexnet13", backbone, pretrained, progress, **kwargs)  # ty: ignore[invalid-argument-type]
```

#### UNetp

```python
UNetp(layout: list[int], in_channels: int = 3, num_classes: int = 10, act_layer: Module | None = None, norm_layer: Callable[[int], Module] | None = None, drop_layer: Callable[..., Module] | None = None, conv_layer: Callable[..., Module] | None = None)
```

Implements a UNet+ architecture

| PARAMETER     | DESCRIPTION                                                             |
| ------------- | ----------------------------------------------------------------------- |
| `layout`      | number of channels after each contracting block **TYPE:** `list[int]`   |
| `in_channels` | number of channels in the input tensor **TYPE:** `int` **DEFAULT:** `3` |
| `num_classes` | number of output classes **TYPE:** `int` **DEFAULT:** `10`              |
| `act_layer`   | activation layer **TYPE:** \`Module                                     |
| `norm_layer`  | normalization layer **TYPE:** \`Callable\[[int], Module\]               |
| `drop_layer`  | dropout layer **TYPE:** \`Callable[..., Module]                         |
| `conv_layer`  | convolutional layer **TYPE:** \`Callable[..., Module]                   |

Source code in `holocron/models/segmentation/unetpp.py`

```python
def __init__(
    self,
    layout: list[int],
    in_channels: int = 3,
    num_classes: int = 10,
    act_layer: nn.Module | None = None,
    norm_layer: Callable[[int], nn.Module] | None = None,
    drop_layer: Callable[..., nn.Module] | None = None,
    conv_layer: Callable[..., nn.Module] | None = None,
) -> None:
    super().__init__()

    if act_layer is None:
        act_layer = nn.ReLU(inplace=True)

    # Contracting path
    self.encoder, self.bridge = _encoder_and_bridge(
        layout, in_channels, act_layer, norm_layer, drop_layer, conv_layer
    )

    # Expansive path
    self.decoder = nn.ModuleList([])
    layout_ = [layout[-1], *layout[1:][::-1]]
    for left_chan, up_chan, num_cells in zip(layout[::-1], layout_, range(1, len(layout) + 1), strict=True):
        self.decoder.append(
            nn.ModuleList([
                UpPath(left_chan + up_chan, left_chan, True, 1, act_layer, norm_layer, drop_layer, conv_layer)
                for _ in range(num_cells)
            ])
        )

    # Classifier
    self.classifier = nn.Conv2d(layout[0], num_classes, 1)

    init_module(self, "relu")
```

#### unetp

```python
unetp(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNetp
```

UNet+ from ["UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation"](https://arxiv.org/pdf/1912.05074.pdf)

| PARAMETER    | DESCRIPTION                                                                                     |
| ------------ | ----------------------------------------------------------------------------------------------- |
| `pretrained` | If True, returns a model pre-trained on PASCAL VOC2012 **TYPE:** `bool` **DEFAULT:** `False`    |
| `progress`   | If True, displays a progress bar of the download to stderr **TYPE:** `bool` **DEFAULT:** `True` |
| `kwargs`     | keyword args of UNetp **TYPE:** `Any` **DEFAULT:** `{}`                                         |

| RETURNS | DESCRIPTION                 |
| ------- | --------------------------- |
| `UNetp` | semantic segmentation model |

Source code in `holocron/models/segmentation/unetpp.py`

```python
def unetp(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNetp:
    """UNet+ from ["UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation"](https://arxiv.org/pdf/1912.05074.pdf)

    ![UNet+ architecture](https://github.com/frgfm/Holocron/releases/download/v0.1.3/unetp.png)

    Args:
        pretrained: If True, returns a model pre-trained on PASCAL VOC2012
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of [`UNetp`][holocron.models.segmentation.unetpp.UNetp]

    Returns:
        semantic segmentation model
    """
    return _unet("unetp", pretrained, progress, **kwargs)  # type: ignore[return-value]
```

#### UNetpp

```python
UNetpp(layout: list[int], in_channels: int = 3, num_classes: int = 10, act_layer: Module | None = None, norm_layer: Callable[[int], Module] | None = None, drop_layer: Callable[..., Module] | None = None, conv_layer: Callable[..., Module] | None = None)
```

Implements a UNet++ architecture

| PARAMETER     | DESCRIPTION                                                             |
| ------------- | ----------------------------------------------------------------------- |
| `layout`      | number of channels after each contracting block **TYPE:** `list[int]`   |
| `in_channels` | number of channels in the input tensor **TYPE:** `int` **DEFAULT:** `3` |
| `num_classes` | number of output classes **TYPE:** `int` **DEFAULT:** `10`              |
| `act_layer`   | activation layer **TYPE:** \`Module                                     |
| `norm_layer`  | normalization layer **TYPE:** \`Callable\[[int], Module\]               |
| `drop_layer`  | dropout layer **TYPE:** \`Callable[..., Module]                         |
| `conv_layer`  | convolutional layer **TYPE:** \`Callable[..., Module]                   |

Source code in `holocron/models/segmentation/unetpp.py`

```python
def __init__(
    self,
    layout: list[int],
    in_channels: int = 3,
    num_classes: int = 10,
    act_layer: nn.Module | None = None,
    norm_layer: Callable[[int], nn.Module] | None = None,
    drop_layer: Callable[..., nn.Module] | None = None,
    conv_layer: Callable[..., nn.Module] | None = None,
) -> None:
    super().__init__()

    if act_layer is None:
        act_layer = nn.ReLU(inplace=True)

    # Contracting path
    self.encoder, self.bridge = _encoder_and_bridge(
        layout, in_channels, act_layer, norm_layer, drop_layer, conv_layer
    )

    # Expansive path
    self.decoder = nn.ModuleList([])
    layout_ = [layout[-1], *layout[1:][::-1]]
    for left_chan, up_chan, num_cells in zip(layout[::-1], layout_, range(1, len(layout) + 1), strict=True):
        self.decoder.append(
            nn.ModuleList([
                UpPath(
                    up_chan + (idx + 1) * left_chan,
                    left_chan,
                    True,
                    1,
                    act_layer,
                    norm_layer,
                    drop_layer,
                    conv_layer,
                )
                for idx in range(num_cells)
            ])
        )

    # Classifier
    self.classifier = nn.Conv2d(layout[0], num_classes, 1)

    init_module(self, "relu")
```

#### unetpp

```python
unetpp(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNetpp
```

UNet++ from ["UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation"](https://arxiv.org/pdf/1912.05074.pdf)

| PARAMETER    | DESCRIPTION                                                                                     |
| ------------ | ----------------------------------------------------------------------------------------------- |
| `pretrained` | If True, returns a model pre-trained on PASCAL VOC2012 **TYPE:** `bool` **DEFAULT:** `False`    |
| `progress`   | If True, displays a progress bar of the download to stderr **TYPE:** `bool` **DEFAULT:** `True` |
| `kwargs`     | keyword args of UNetpp **TYPE:** `Any` **DEFAULT:** `{}`                                        |

| RETURNS  | DESCRIPTION                 |
| -------- | --------------------------- |
| `UNetpp` | semantic segmentation model |

Source code in `holocron/models/segmentation/unetpp.py`

```python
def unetpp(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNetpp:
    """UNet++ from ["UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation"](https://arxiv.org/pdf/1912.05074.pdf)

    ![UNet++ architecture](https://github.com/frgfm/Holocron/releases/download/v0.1.3/unetpp.png)

    Args:
        pretrained: If True, returns a model pre-trained on PASCAL VOC2012
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of [`UNetpp`][holocron.models.segmentation.unetpp.UNetpp]

    Returns:
        semantic segmentation model
    """
    return _unet("unetpp", pretrained, progress, **kwargs)  # type: ignore[return-value]
```

#### UNet3p

```python
UNet3p(layout: list[int], in_channels: int = 3, num_classes: int = 10, act_layer: Module | None = None, norm_layer: Callable[[int], Module] | None = None, drop_layer: Callable[..., Module] | None = None, conv_layer: Callable[..., Module] | None = None)
```

Implements a UNet3+ architecture

| PARAMETER     | DESCRIPTION                                                             |
| ------------- | ----------------------------------------------------------------------- |
| `layout`      | number of channels after each contracting block **TYPE:** `list[int]`   |
| `in_channels` | number of channels in the input tensor **TYPE:** `int` **DEFAULT:** `3` |
| `num_classes` | number of output classes **TYPE:** `int` **DEFAULT:** `10`              |
| `act_layer`   | activation layer **TYPE:** \`Module                                     |
| `norm_layer`  | normalization layer **TYPE:** \`Callable\[[int], Module\]               |
| `drop_layer`  | dropout layer **TYPE:** \`Callable[..., Module]                         |
| `conv_layer`  | convolutional layer **TYPE:** \`Callable[..., Module]                   |

Source code in `holocron/models/segmentation/unet3p.py`

```python
def __init__(
    self,
    layout: list[int],
    in_channels: int = 3,
    num_classes: int = 10,
    act_layer: nn.Module | None = None,
    norm_layer: Callable[[int], nn.Module] | None = None,
    drop_layer: Callable[..., nn.Module] | None = None,
    conv_layer: Callable[..., nn.Module] | None = None,
) -> None:
    super().__init__()

    if act_layer is None:
        act_layer = nn.ReLU(inplace=True)
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d

    # Contracting path
    self.encoder = nn.ModuleList([])
    layout_ = [in_channels, *layout]
    pool = False
    for in_chan, out_chan in pairwise(layout_):
        self.encoder.append(down_path(in_chan, out_chan, pool, 1, act_layer, norm_layer, drop_layer, conv_layer))
        pool = True

    # Expansive path
    self.decoder = nn.ModuleList([])
    for row in range(len(layout) - 1):
        self.decoder.append(
            FSAggreg(
                layout[:row],
                layout[row],
                [len(layout) * layout[0]] * (len(layout) - 2 - row) + layout[-1:],
                act_layer,
                norm_layer,
                drop_layer,
                conv_layer,
            )
        )

    # Classifier
    self.classifier = nn.Conv2d(len(layout) * layout[0], num_classes, 1)

    init_module(self, "relu")
    # The classifier has no ReLU; keep logit variance independent of the number of classes.
    nn.init.kaiming_normal_(self.classifier.weight, mode="fan_in", nonlinearity="linear")
```

#### unet3p

```python
unet3p(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNet3p
```

UNet3+ from ["UNet 3+: A Full-Scale Connected UNet For Medical Image Segmentation"](https://arxiv.org/pdf/2004.08790.pdf)

| PARAMETER    | DESCRIPTION                                                                                     |
| ------------ | ----------------------------------------------------------------------------------------------- |
| `pretrained` | If True, returns a model pre-trained on PASCAL VOC2012 **TYPE:** `bool` **DEFAULT:** `False`    |
| `progress`   | If True, displays a progress bar of the download to stderr **TYPE:** `bool` **DEFAULT:** `True` |
| `kwargs`     | keyword args of UNet3p **TYPE:** `Any` **DEFAULT:** `{}`                                        |

| RETURNS  | DESCRIPTION                 |
| -------- | --------------------------- |
| `UNet3p` | semantic segmentation model |

Source code in `holocron/models/segmentation/unet3p.py`

```python
def unet3p(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNet3p:
    """UNet3+ from
    ["UNet 3+: A Full-Scale Connected UNet For Medical Image Segmentation"](https://arxiv.org/pdf/2004.08790.pdf)

    ![UNet 3+ architecture](https://github.com/frgfm/Holocron/releases/download/v0.1.3/unet3p.png)

    Args:
        pretrained: If True, returns a model pre-trained on PASCAL VOC2012
        progress: If True, displays a progress bar of the download to stderr
        kwargs: keyword args of [`UNet3p`][holocron.models.segmentation.unet3p.UNet3p]

    Returns:
        semantic segmentation model
    """
    return _unet("unet3p", pretrained, progress, **kwargs)  # type: ignore[return-value]
```

### YOLO26 family

See [YOLO26 nano semantic segmentation](../segmentation/yolo26/) for the model API, training output, parameter counts, and deployment limits.
