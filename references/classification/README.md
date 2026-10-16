# Image classification

Since I do not own enough computing power to iterate over ImageNet full training, this section involves training on a subset of ImageNet, called [Imagenette](https://github.com/fastai/imagenette).

## Getting started

Follow the [shared installation instructions](../README.md#installation). Run the commands below from `references/classification`.

Download [Imagenette](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz) and extract it where you want

```bash
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xvzf imagenette2-320.tgz
```

From there, you can run your training with the following command

```
uv run --project ../.. --no-sync python train.py imagenette2-320/ --arch darknet53 --lr 5e-3 -b 32 -j 16 --epochs 40 --opt adamp --sched onecycle
```

## Synthetic character classification

From the repository root, prepare the checksum-verified starter fonts once:

```shell
uv run --python 3.12 scripts/prepare_fonts.py \
  --output /tmp/holocron-fonts --quiet
```

Generate and inspect fresh augmented samples without starting training:

```shell
uv run --python 3.12 --extra training \
  references/classification/train_characters.py \
  --font-dir /tmp/holocron-fonts \
  --manifest references/fonts/latin-starter.json \
  --show-samples /tmp/character-samples.png
```

Run a small CPU smoke training job:

```shell
uv run --python 3.12 --extra training \
  references/classification/train_characters.py \
  --font-dir /tmp/holocron-fonts \
  --manifest references/fonts/latin-starter.json \
  --device cpu --workers 0 --epochs 1 \
  --image-size 32 --render-size 64 \
  --samples-per-epoch 256 --batch-size 32
```

Measure live-image DataLoader throughput after worker warm-up:

```shell
uv run --python 3.12 --extra training \
  references/classification/train_characters.py \
  --font-dir /tmp/holocron-fonts \
  --manifest references/fonts/latin-starter.json \
  --benchmark-loader
```

`--render-size` controls glyph rasterization resolution; `--image-size` controls the final model input resolution. To expand the corpus, prepare another local manifest and directory and pass those paths to the same script—no Python changes are needed. Training itself never downloads data or fonts.

`--arch alexnet` reuses TorchVision's AlexNet feature stack with a compact task head; use image size 64 or larger.

## Personal leaderboard

The updated list of available checkpoints can be found in the [documentation](https://frgfm.github.io/holocron/reference/models/models/#available-checkpoints).


## Imagenette

| Model            | Accuracy@1 (Err) | Param # | MACs  | Interpolation | Image size |
| ---------------- | ---------------- | ------- | ----- | ------------- | ---------- |
| cspdarknet53     | 92.54 (7.46)     | 26.63M  | 5.03G | bilinear      | 224        |
| cspdarknet53_mish| 94.14 (5.86)     | 26.63M  | 5.03G | bilinear      | 256        |
| rexnet2_2x       | 91.75 (8.25)     | 19.49M  | 1.88G | bilinear      | 224        |
| rexnet50d        | 92.18 (7.82)     | 23.55M  | 4.35G | bilinear      | 224        |
| darknet53        | 91.46 (8.54)     | 40.60M  | 9.31G | bilinear      | 256        |
| repvgg_a2        | 91.26 (8.74)     | 48.63M  |       | bilinear      | 224        |
| darknet19        | 91.87 (8.13)     | 19.83M  | 2.75G | bilinear      | 224        |
| tridentresnet50  | 91.01 (8.99)     | 45.83M  | 35.9G | bilinear      | 224        |
| sknet50          | 90.42 (9.58)     | 35.22M  | 5.96G | bilinear      | 224        |
| rexnet1_3x       | 94.06 (5.94)     | 7.56M   | 0.68G | bilinear      | 224        |
| repvgg_a1        | 90.97 (9.03)     | 30.12M  |       | bilinear      | 224        |
| rexnet1_0x       | 92.99 (7.01)     | 4.80M   | 0.42G | bilinear      | 224        |
| repvgg_a0        | 91.18 (8.82)     | 24.74M  |       | bilinear      | 224        |
| repvgg_b0        | 89.61 (9.39)     | 31.85M  |       | bilinear      | 224        |
| res2net50_26w_4s | 89.58 (99.26)    | 23.67M  | 4.28G | bilinear      | 224        |
| darnet24         | 91.57 (8.43)     | 22.40M  | 4.21G | bilinear      | 224        |
| resnet50         | 84.36 (15.64)    | 23.53M  | 4.11G | bilinear      | 224        |

For character pretraining that transfers directly into line and page OCR, see the [recognition experiment](../recognition/README.md).
