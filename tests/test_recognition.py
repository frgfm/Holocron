import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from holocron.models.recognition import CharacterClassifier, CTCRecognizer
from holocron.utils import CTCCodec, prefix_beam_decode
from references.classification.train_characters import FontRecord
from references.recognition import evaluate, train
from references.recognition.data import (
    ALPHABET,
    SyntheticTextDataset,
    collate_lines,
    deskew_line,
    image_tensor,
    inspect_fonts,
    render_line,
    render_page,
    split_fonts,
)
from references.recognition.evaluate import (
    detect_lines,
    edit_distance,
    load_checkpoint,
    transcribe_page,
    transcript_metrics,
)

SANS = Path(mpl.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
SERIF = SANS.with_name("DejaVuSerif.ttf")


@pytest.fixture
def records():
    return inspect_fonts([SANS, SERIF], CTCCodec(ALPHABET).alphabet)


def test_ctc_repeats_spaces_punctuation_and_impossible_alignment():
    codec = CTCCodec("AB .")
    assert codec.decode([1, 1, 0, 1, 3, 3, 2, 4, 4]) == "AA B."
    assert not codec.decode([0, 0])
    probabilities = torch.randn(5, 1, 5).log_softmax(-1).requires_grad_()
    loss = train.ctc_loss(probabilities, torch.tensor([5]), ["AAB"], codec)
    loss.backward()
    assert torch.isfinite(loss)
    assert probabilities.grad is not None
    with pytest.raises(ValueError, match="repeated"):
        train.ctc_loss(probabilities, torch.tensor([2]), ["AA"], codec)


@pytest.mark.parametrize("alphabet", ["", "AA", "A\n", " "])
def test_invalid_alphabet(alphabet):
    with pytest.raises(ValueError):
        CTCCodec(alphabet)


def test_family_split_and_missing_glyph_validation(records):
    fonts = (*records, FontRecord(SANS, records[0].family, "Another style"))
    seen, unseen = split_fonts(fonts, [records[0].family])
    assert len(unseen) == 2
    assert {font.family for font in seen}.isdisjoint(font.family for font in unseen)
    with pytest.raises(ValueError, match="unknown"):
        split_fonts(fonts, ["Missing"])
    with pytest.raises(ValueError, match="complete alphabet"):
        inspect_fonts([SANS], "A你")


def test_data_seed_epoch_and_worker_independence(records):
    codec = CTCCodec("AB12 .")
    dataset = SyntheticTextDataset(records, codec, 8, seed=42, augment=True)
    first = [dataset[index] for index in range(8)]
    for index in reversed(range(8)):
        image, text = dataset[index]
        assert text == first[index][1]
        assert torch.equal(image, first[index][0])
        assert image.dtype == torch.float32
        assert image.shape[-2] == 32
        assert image.shape[-1] % 4 == 0
    plain = list(DataLoader(dataset, batch_size=4, collate_fn=collate_lines, num_workers=0))
    workers = list(DataLoader(dataset, batch_size=4, collate_fn=collate_lines, num_workers=1))
    for a, b in zip(plain, workers, strict=True):
        assert torch.equal(a[0], b[0])
        assert torch.equal(a[1], b[1])
        assert a[2] == b[2]
    dataset.epoch = 1
    assert not torch.equal(dataset[0][0], first[0][0])


def test_variable_width_padding_and_masked_context(records):
    dataset = SyntheticTextDataset(records, CTCCodec(ALPHABET), 5)
    images, lengths, texts = collate_lines([dataset[index] for index in range(5)])
    assert images.shape[-1] == int(lengths.max()) * 4
    for index, length in enumerate(lengths):
        assert (images[index, :, :, int(length) * 4 :] == 1).all()
    model = CTCRecognizer(len(CTCCodec(ALPHABET).alphabet)).eval()
    probabilities = model(images, lengths)
    assert probabilities.shape == (int(lengths.max()), 5, len(CTCCodec(ALPHABET).alphabet) + 1)
    loss = train.ctc_loss(probabilities, lengths, texts, CTCCodec(ALPHABET))
    loss.backward()
    assert model.backbone.features[0].weight.grad.abs().sum() > 0


def test_metrics_include_insertions_deletions_and_whitespace():
    assert edit_distance("book", "bok") == 1
    assert edit_distance("", "abc") == 3
    metrics = transcript_metrics(["A B", "CC"], ["AB", "CCC"])
    assert metrics["cer"] == 2 / 5
    assert metrics["exact_match"] == 0
    assert metrics["word_errors"] == 3
    assert transcript_metrics([""], ["A"])["cer"] == 1
    with pytest.raises(ValueError):
        transcript_metrics(["A"], [])


def test_page_detection_and_reading_order_use_only_pixels(records):
    codec = CTCCodec(ALPHABET)
    for degraded in (False, True):
        page = render_page(records, codec, 17, augment=degraded)
        boxes = detect_lines(page.image)
        assert len(boxes) == len(page.text.splitlines())
        assert [box[1] for box in boxes] == sorted(box[1] for box in boxes)
        assert detect_lines(Image.new("L", (100, 100), 255)) == []
    assert render_page(records, codec, 17).text == render_page(records, codec, 17, augment=True).text
    model = CTCRecognizer(len(codec.alphabet)).eval()
    text, boxes = transcribe_page(model, codec, page.image)
    assert len(text.split("\n")) == len(boxes)
    blank_text, blank_boxes = transcribe_page(model, codec, Image.new("L", (100, 100), 255))
    assert not blank_text
    assert blank_boxes == []


def test_page_can_handle_inverted_pixels():
    image = Image.new("L", (120, 100), 0)
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 90, 20), fill=255)
    draw.rectangle((20, 55, 100, 70), fill=255)
    assert len(detect_lines(image)) == 2


def test_real_cli_pretraining_transfer_resume_and_evaluation(tmp_path):
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    for path in (SANS, SERIF):
        (font_dir / path.name).write_bytes(path.read_bytes())
    common = [
        "--font-dir",
        str(font_dir),
        "--held-out",
        "DejaVu Serif",
        "--alphabet",
        "AB12 .",
        "--epochs",
        "1",
        "--samples-per-epoch",
        "8",
        "--validation-samples",
        "4",
        "--batch-size",
        "4",
        "--workers",
        "0",
        "--threads",
        "1",
    ]
    char_dir, line_dir = tmp_path / "characters", tmp_path / "lines"
    train.main(train.get_parser().parse_args([*common, "--task", "characters", "--output-dir", str(char_dir)]))
    model, _codec, checkpoint = load_checkpoint(char_dir / "best.pth", "cpu")
    assert isinstance(model, CharacterClassifier)
    assert checkpoint["fonts"][0]["sha256"]
    line_args = [*common, "--output-dir", str(line_dir), "--init", str(char_dir / "best.pth")]
    train.main(train.get_parser().parse_args(line_args))
    model, _codec, checkpoint = load_checkpoint(line_dir / "best.pth", "cpu")
    assert isinstance(model, CTCRecognizer)
    assert checkpoint["initialized_from"]["task"] == "characters"
    assert checkpoint["history"][0]["train_loss"] > 0
    result_path = tmp_path / "evaluation.json"
    evaluate.main(
        evaluate.get_parser().parse_args([
            str(line_dir / "best.pth"),
            "--font-dir",
            str(font_dir),
            "--samples",
            "4",
            "--pages",
            "2",
            "--threads",
            "1",
            "--output",
            str(result_path),
        ])
    )
    results = json.loads(result_path.read_text())["benchmarks"]
    assert results["seen_clean"]["samples"] == 4
    assert results["unseen_degraded_pages"]["samples"] == 2
    assert results["unseen_clean"]["families"] == ["DejaVu Serif"]
    # Initialization must preserve the family holdout across curriculum stages.
    leaked = torch.load(char_dir / "best.pth", weights_only=True)
    leaked["fonts"].append({"family": "DejaVu Serif"})
    leaked_path = tmp_path / "leaked.pth"
    torch.save(leaked, leaked_path)
    with pytest.raises(ValueError, match="held-out"):
        train.main(train.get_parser().parse_args([*common, "--output-dir", str(line_dir), "--init", str(leaked_path)]))
    train.main(
        train.get_parser().parse_args([*common, "--output-dir", str(line_dir), "--resume", str(line_dir / "last.pth")])
    )
    with pytest.raises(ValueError, match="configuration"):
        train.main(
            train.get_parser().parse_args([
                *common,
                "--lr",
                "0.01",
                "--output-dir",
                str(line_dir),
                "--resume",
                str(line_dir / "last.pth"),
            ])
        )
    metadata = json.loads((line_dir / "history.json").read_text())
    assert {font["family"] for font in metadata["fonts"]}.isdisjoint(
        font["family"] for font in metadata["held_out_fonts"]
    )


def test_small_fixed_batch_learns_character_shapes(records):
    torch.manual_seed(0)
    dataset = SyntheticTextDataset(records[:1], CTCCodec("AB"), 2, task="characters")
    images, lengths, _ = collate_lines([dataset[0], dataset[1]])
    model = CharacterClassifier(2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    target = torch.tensor([0, 1])
    initial = torch.nn.functional.cross_entropy(model(images, lengths), target).item()
    for _ in range(15):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(images, lengths), target)
        loss.backward()
        optimizer.step()
    assert loss.item() < initial / 3


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_resume_matches_uninterrupted_optimization(monkeypatch, tmp_path, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip(f"{device} unavailable")
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    for path in (SANS, SERIF):
        (font_dir / path.name).write_bytes(path.read_bytes())
    common = [
        "--device",
        device,
        "--font-dir",
        str(font_dir),
        "--held-out",
        "DejaVu Serif",
        "--alphabet",
        "AB12 .",
        "--epochs",
        "2",
        "--samples-per-epoch",
        "8",
        "--validation-samples",
        "4",
        "--batch-size",
        "4",
        "--workers",
        "0",
        "--threads",
        "1",
    ]
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    saved = tmp_path / "epoch-one.pth"
    original_save = torch.save

    def capture_checkpoint(checkpoint, path):
        original_save(checkpoint, path)
        if checkpoint["epoch"] == 1 and Path(path).name == "last.pth":
            original_save(checkpoint, saved)

    monkeypatch.setattr(train.torch, "save", capture_checkpoint)
    train.main(train.get_parser().parse_args([*common, "--output-dir", str(continuous)]))
    monkeypatch.setattr(train.torch, "save", original_save)
    train.main(train.get_parser().parse_args([*common, "--output-dir", str(resumed), "--resume", str(saved)]))
    expected = torch.load(continuous / "last.pth", weights_only=True)
    actual = torch.load(resumed / "last.pth", weights_only=True)
    for key, value in expected["model"].items():
        assert torch.equal(value, actual["model"][key])
    assert expected["history"][-1]["train_loss"] == actual["history"][-1]["train_loss"]
    assert expected["scheduler"] == actual["scheduler"]


def test_character_preprocessing_preserves_punctuation_baseline(records):
    hyphen = image_tensor(render_line("-", records[0], random.Random(0)), character=True)  # noqa: S311
    underscore = image_tensor(render_line("_", records[0], random.Random(0)), character=True)  # noqa: S311
    assert hyphen.shape == underscore.shape
    assert not torch.equal(hyphen, underscore)
    # Classification must retain the vertical location, even for tiny glyphs.
    hyphen_rows = (hyphen[0] < 0).nonzero()[:, 0].float().mean()
    underscore_rows = (underscore[0] < 0).nonzero()[:, 0].float().mean()
    assert underscore_rows > hyphen_rows + 5


def test_prefix_beam_matches_exhaustive_ctc_alignment():
    codec = CTCCodec("AB")
    probabilities = np.array([[0.45, 0.4, 0.15]] * 3)
    assert not codec.decode(probabilities.argmax(-1))
    assert prefix_beam_decode(np.log(probabilities), codec) == "A"
    probabilities = np.array([[0.1, 0.8, 0.1], [0.6, 0.3, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]])
    transcript_probabilities = defaultdict(float)
    for alignment in itertools.product(range(3), repeat=4):
        transcript_probabilities[codec.decode(alignment)] += np.prod([
            probabilities[index, token] for index, token in enumerate(alignment)
        ])
    expected = max(transcript_probabilities, key=transcript_probabilities.get)
    assert prefix_beam_decode(np.log(probabilities), codec, beam_width=64) == expected
    with pytest.raises(ValueError, match="positive"):
        prefix_beam_decode(np.log(probabilities), codec, beam_width=0)


def test_deskew_improves_long_rotated_lines_and_preserves_blank_short_images(records):
    font = ImageFont.truetype(str(records[0].path), 28)
    image = Image.new("L", (500, 60), 255)
    ImageDraw.Draw(image).text((10, 8), "The quick brown fox jumps 12345", font=font, fill=0)
    rotated = image.rotate(1.5, Image.Resampling.BILINEAR, expand=True, fillcolor=255)
    corrected = deskew_line(rotated)

    def row_concentration(candidate):
        counts = (np.asarray(candidate) < 180).sum(1).astype(float)
        return (counts**2).sum() / counts.sum()

    assert row_concentration(corrected) >= row_concentration(rotated) * 1.03
    assert deskew_line(Image.new("L", (500, 32), 255)).getextrema() == (255, 255)
    short = Image.new("L", (32, 32), 255)
    assert deskew_line(short) is short
    tensor = image_tensor(rotated, deskew=True)
    assert tensor.shape[-2] == 32
    assert tensor.shape[-1] % 4 == 0


def test_staged_schedule_can_continue_after_an_early_stop(tmp_path):
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    for path in (SANS, SERIF):
        (font_dir / path.name).write_bytes(path.read_bytes())
    output = tmp_path / "staged"
    common = [
        "--font-dir",
        str(font_dir),
        "--held-out",
        "DejaVu Serif",
        "--alphabet",
        "AB12 .",
        "--epochs",
        "2",
        "--samples-per-epoch",
        "4",
        "--validation-samples",
        "4",
        "--batch-size",
        "4",
        "--workers",
        "0",
        "--threads",
        "1",
        "--deskew",
        "--output-dir",
        str(output),
    ]
    train.main(train.get_parser().parse_args([*common, "--stop-after-epochs", "1"]))
    assert torch.load(output / "last.pth", weights_only=True)["epoch"] == 1
    train.main(train.get_parser().parse_args([*common, "--resume", str(output / "last.pth")]))
    assert torch.load(output / "last.pth", weights_only=True)["epoch"] == 2
    model, _, _ = load_checkpoint(output / "best.pth", "cpu")
    assert model.deskew


def test_page_detector_keeps_thin_single_glyphs_and_punctuation_fragments():
    image = Image.new("L", (180, 180), 255)
    draw = ImageDraw.Draw(image)
    draw.line((20, 15, 20, 30), fill=190, width=1)
    draw.rectangle((20, 60, 22, 62), fill=0)
    draw.rectangle((20, 73, 22, 75), fill=0)
    draw.rectangle((15, 108, 150, 127), fill=0)
    boxes = detect_lines(image)
    assert len(boxes) == 3
    assert boxes[0][1] <= 15
    assert boxes[0][3] > 30
    assert boxes[1][1] <= 60
    assert boxes[1][3] > 75
