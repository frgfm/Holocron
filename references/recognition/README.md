# From character classification to synthetic OCR

Measured accuracy, training histories, and error examples are in [RESULTS.md](RESULTS.md).

This experiment extends [synthetic character classification](../classification/README.md#synthetic-character-classification) with a shared visual backbone, character pretraining, variable-width line recognition, and page transcription. It is a reproducible research reference, using only existing Holocron dependencies. It does not download pretrained OCR weights or use a dictionary to correct predictions.

The reusable models are part of the installed library: `holocron.models.recognition.CharacterBackbone`, `CharacterClassifier`, and `CTCRecognizer`. Alphabet encoding and greedy/prefix-beam decoding use `holocron.utils.CTCCodec` and `prefix_beam_decode`. Their [public API documentation](../../docs/docs/reference/models/recognition.md) describes input shapes, normalization, and sequence lengths. This folder contains synthetic rendering, augmentation, training orchestration, evaluation, and the experimental page detector; it imports the library models directly.

The curriculum has three stages:

1. Classify balanced synthetic glyphs across font families, keeping their baseline position so punctuation remains distinguishable.
2. Transfer the CNN into a two-layer bidirectional GRU and train with CTC on lines mixing arbitrary strings and document fields. CTC learns character alignment without character boxes, including repeated characters and spaces.
3. Detect separated horizontal lines from page pixels, recognize them, and assemble a transcript in top-to-bottom order. Synthetic boxes are used for scoring line counts only.

The model has about 0.50 million parameters. Its 83-symbol alphabet contains 82 visible ASCII characters and ordinary space. Images are grayscale, height 32, with aspect ratio preserved. Both CTC loss and the packed recurrent layers use each line's true time length; batch padding is excluded from decoding. The classifier retains vertical glyph position; line crops normalize the ink bounding box for compatibility with detected page lines.

## Reproduce the experiment

Run these commands from the repository root in its installed environment (`uv sync --locked`). Training itself performs no downloads. The corpus manifest pins the Google Fonts revision, individual SHA-256 checksums, and font licenses.

```shell
uv run --no-sync python scripts/prepare_fonts.py \
  --manifest references/fonts/ocr-latin.json --output /tmp/ocr-fonts --quiet

# Stage 1: balanced character pretraining.
uv run --no-sync python -m references.recognition.train \
  --task characters --font-dir /tmp/ocr-fonts \
  --manifest references/fonts/ocr-latin.json \
  --epochs 16 --samples-per-epoch 8192 --validation-samples 1024 \
  --batch-size 64 --threads 2 --workers 1 \
  --output-dir checkpoints/ocr/characters

# Control: sequence training from scratch, without augmentation.
uv run --no-sync python -m references.recognition.train \
  --font-dir /tmp/ocr-fonts --manifest references/fonts/ocr-latin.json \
  --no-augment --epochs 20 --samples-per-epoch 2048 \
  --validation-samples 256 --threads 2 --workers 1 \
  --output-dir checkpoints/ocr/baseline

# Stage 2: transfer + augmentation, with the same sequence training budget.
uv run --no-sync python -m references.recognition.train \
  --font-dir /tmp/ocr-fonts --manifest references/fonts/ocr-latin.json \
  --init checkpoints/ocr/characters/best.pth \
  --epochs 20 --samples-per-epoch 2048 --validation-samples 256 \
  --threads 2 --workers 1 --output-dir checkpoints/ocr/transfer

# Independent line and page evaluation.
for run in baseline transfer; do
  uv run --no-sync python -m references.recognition.evaluate \
    checkpoints/ocr/$run/best.pth \
    --font-dir /tmp/ocr-fonts --manifest references/fonts/ocr-latin.json \
    --samples 2000 --pages 200 --seed 400000 --threads 1 \
    --output checkpoints/ocr/$run/evaluation.json
done

uv run --no-sync python -m references.recognition.evaluate \
  checkpoints/ocr/characters/best.pth \
  --font-dir /tmp/ocr-fonts --manifest references/fonts/ocr-latin.json \
  --samples 8192 --pages 0 --seed 300000 --threads 1 \
  --output checkpoints/ocr/characters/evaluation.json
```

The default split trains on Montserrat, Lora, Roboto Mono, Noto Sans, Open Sans, Source Sans 3, IBM Plex Mono, and PT Serif; all Noto Serif and Ubuntu images are held out. Splitting uses inspected family names, keeping all styles of each family together. Character pretraining and sequence training use only the training families. Initialization rejects a checkpoint trained on a held-out family. Unsupported glyphs and fonts that map lowercase to uppercase are rejected.

Training uses AdamW, cosine learning-rate decay, gradient clipping, and greedy CTC decoding. Augmentation varies font size, horizontal scale, rotation, blur, contrast, and noise. Each sample has its own seed derived from the training seed, index, and epoch, so rendering is reproducible across worker counts and access order. Sampling chooses a family before a style, preventing families with many styles from dominating.

`history.json` records losses, validation CER, exact matches, timing, arguments, font checksums, initialization provenance, and runtime versions. `best.pth` is selected by a fixed, degraded validation set drawn only from training families, using the training seed plus 100000; `last.pth` additionally supports exact continuation. Evaluation uses an independent seed, records runtime versions, checkpoint and font hashes, and reports clean/degraded results separately for seen/unseen families. CER and WER are micro-averaged edit distance divided by the total reference characters/words; exact match counts entirely correct strings. Page CER includes newlines, so missing lines affect transcript scoring.

To resume an interrupted run, repeat its configuration and add `--resume checkpoints/ocr/transfer/last.pth`. Keep the total epoch count unchanged. Optimizer, scheduler, training history, and PyTorch RNG are restored; changed training settings or font checksums are rejected. Switching backends is supported, but does not promise identical RNG streams or results. To deliberately start another training phase with a different budget or line length, use `--init` instead.

## Longer-line curriculum and normalization

For the longer-line refinement, warm-start the transfer model with a smaller learning rate and fresh seeds:

```shell
uv run --no-sync python -m references.recognition.train \
  --font-dir /tmp/ocr-fonts --manifest references/fonts/ocr-latin.json \
  --init checkpoints/ocr/transfer/best.pth \
  --epochs 24 --stop-after-epochs 13 --samples-per-epoch 2048 --validation-samples 512 \
  --max-length 48 --seed 1 --lr 0.0003 --threads 3 --workers 1 \
  --output-dir checkpoints/ocr/long-lines
```

`--beam-width 5` enables prefix beam search in evaluation or image transcription. It tracks blank and nonblank paths separately, allowing repeated characters while summing alternative alignments. For bounded CPU cost it retains the top eight nonblank tokens per frame. This is an approximate decoder with no dictionary or language model. Greedy decoding remains the default and the primary controlled comparison.

The measured raw longer-line phase ends after 13 epochs of its 24-epoch cosine schedule. It was followed by training on deskewed inputs after validation showed that small rotations were a major source of errors:

```shell
uv run --no-sync python -m references.recognition.train \
  --font-dir /tmp/ocr-fonts --manifest references/fonts/ocr-latin.json \
  --init checkpoints/ocr/long-lines/best.pth \
  --epochs 12 --samples-per-epoch 4096 --validation-samples 512 \
  --max-length 48 --seed 2 --lr 0.0003 --threads 3 --workers 1 --deskew \
  --output-dir checkpoints/ocr/deskew
```

Deskewing searches corrections from -2 to +2 degrees at 0.5-degree intervals. It applies only to wide line crops and requires at least a 3% improvement in horizontal ink concentration. Character pretraining keeps its original baseline. The setting is saved in the checkpoint and applied automatically during evaluation and page transcription; `--deskew` / `--no-deskew` explicitly override it for ablations. The default for the earlier stages remains disabled.

Evaluate the final checkpoint on the same fresh-seed short-line/page benchmark, then on the longer-line stress benchmark:

```shell
uv run --no-sync python -m references.recognition.evaluate \
  checkpoints/ocr/deskew/best.pth \
  --font-dir /tmp/ocr-fonts --manifest references/fonts/ocr-latin.json \
  --samples 2000 --pages 200 --seed 400000 --threads 2 \
  --output checkpoints/ocr/deskew/evaluation.json

uv run --no-sync python -m references.recognition.evaluate \
  checkpoints/ocr/deskew/best.pth \
  --font-dir /tmp/ocr-fonts --manifest references/fonts/ocr-latin.json \
  --max-length 48 --samples 1000 --pages 100 --seed 500000 --threads 2 \
  --output checkpoints/ocr/deskew/long-evaluation.json
```

Regenerate the checked-in learning curves with `python -m references.recognition.plot_results`.

## Transcribe an image

```shell
uv run --no-sync python -m references.recognition.evaluate \
  checkpoints/ocr/deskew/best.pth \
  --image /path/to/page.png --output /tmp/transcript.json
```

The result contains the transcript and inferred line boxes. The same `transcribe_page` function used by the benchmark handles user-supplied images. It uses no font metadata or synthetic annotations at inference time.

## Scope and next experiments

The page detector assumes separated, horizontal, single-column text on a plain background. It does not handle tables, multiple columns, illustrations, touching lines, or strongly skewed scans. Page degradation currently consists of independently degraded text lines on a white canvas. It is not a real scanned-document benchmark. The renderer generates fields and random strings up to 24 characters in the controlled comparison and 48 in the longer-line curriculum, rather than long natural-language paragraphs. Sparse punctuation-only crops can lose baseline clues; visually identical glyphs also impose an ambiguity ceiling. Unsupported Unicode, leading/trailing whitespace, handwriting, and multilingual reading order are outside the measured scope.

Follow-on experiments should extend the line-length curriculum, font styles and weights, scan/background degradation, and document layouts; replace projection detection with a trained detector; and evaluate on a licensed real-document dataset before claiming production OCR accuracy. The shared backbone and length-aware CTC pipeline make those extensions possible without starting over.
