import gc
import json
import shutil
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import pytest
import torch
from PIL import ImageFont
from torch import nn
from torch.utils.data import SequentialSampler
from torchvision.transforms import v2 as T

from references.classification import train_characters as characters

FONT_ROOT = Path(mpl.get_data_path()) / "fonts" / "ttf"
SANS = FONT_ROOT / "DejaVuSans.ttf"


def _transform_args(*args):
    return characters.get_parser().parse_args([
        "--image-size",
        "24",
        "--render-size",
        "32",
        "--margin",
        "2",
        *args,
    ])


def test_dataset_balance_outputs_and_reproducibility():
    train_transform, val_transform = characters.build_transforms(
        _transform_args("--rotation", "20", "--translate", "0.2", "--scale-jitter", "0.2")
    )
    record = characters.FontRecord(SANS, "DejaVu Sans")
    dataset = characters.SyntheticCharacterDataset("ABC", [record], 32, 7, train_transform)

    assert dataset.classes == ("A", "B", "C")
    assert dataset.class_to_idx == {"A": 0, "B": 1, "C": 2}
    assert Counter(dataset[index][1] for index in range(len(dataset))) == {0: 3, 1: 2, 2: 2}
    image, target = dataset[1]
    assert image.shape == (3, 24, 24)
    assert image.dtype == torch.float32
    assert target == 1

    torch.manual_seed(3)
    first = dataset[0][0]
    torch.manual_seed(3)
    repeated = dataset[0][0]
    torch.manual_seed(4)
    different = dataset[0][0]
    assert torch.equal(first, repeated)
    assert not torch.equal(first, different)

    validation = characters.SyntheticCharacterDataset("ABC", [record], 32, 6, val_transform, deterministic=True)
    first_pass = [validation[index] for index in range(len(validation))]
    second_pass = [validation[index] for index in range(len(validation))]
    assert [target for _, target in first_pass] == [0, 1, 2, 0, 1, 2]
    assert all(torch.equal(first[0], second[0]) for first, second in zip(first_pass, second_pass, strict=True))


def test_family_is_sampled_before_style():
    records = [
        characters.FontRecord(FONT_ROOT / "DejaVuSans.ttf", "Sans", "Regular"),
        characters.FontRecord(FONT_ROOT / "DejaVuSans-Bold.ttf", "Sans", "Bold"),
        characters.FontRecord(FONT_ROOT / "DejaVuSans-Oblique.ttf", "Sans", "Oblique"),
        characters.FontRecord(FONT_ROOT / "DejaVuSerif.ttf", "Serif", "Regular"),
    ]
    to_tensor = T.PILToTensor()
    expected = {}
    for record in records:
        font = ImageFont.truetype(str(record.path), 32)
        image = to_tensor(characters.render_text("A", font))
        expected[tuple(image.shape), image.numpy().tobytes()] = (record.family, record.style)
    assert len(expected) == len(records)

    dataset = characters.SyntheticCharacterDataset("A", records, 32, 1, to_tensor)
    assert {record.family for record in dataset.fonts} == {"DejaVu Sans", "DejaVu Serif"}
    torch.manual_seed(0)
    counts = Counter()
    for _ in range(1_000):
        image, _ = dataset[0]
        counts[expected[tuple(image.shape), image.numpy().tobytes()]] += 1

    sans_count = sum(count for (family, _), count in counts.items() if family == "Sans")
    assert 430 < sans_count < 570
    for style in ("Regular", "Bold", "Oblique"):
        assert 0.25 < counts["Sans", style] / sans_count < 0.42


@pytest.mark.parametrize(
    ("alphabet", "message"),
    [
        ("", "must not be empty"),
        ("AA", "duplicate"),
        (" ", "visible"),
        ("你", "no visible font supports"),
    ],
)
def test_dataset_rejects_invalid_alphabets(alphabet, message):
    with pytest.raises(ValueError, match=message):
        characters.SyntheticCharacterDataset(alphabet, [SANS], 32, 1, T.PILToTensor())


def test_dataset_rejects_invalid_fonts(tmp_path):
    missing = tmp_path / "missing.ttf"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        characters.SyntheticCharacterDataset("A", [missing], 32, 1, T.PILToTensor())

    corrupt = tmp_path / "corrupt.ttf"
    corrupt.write_text("not a font", encoding="utf-8")
    with pytest.raises(ValueError, match="unable to load"):
        characters.SyntheticCharacterDataset("A", [corrupt], 32, 1, T.PILToTensor())


def test_font_objects_are_reused(monkeypatch):
    dataset = characters.SyntheticCharacterDataset("A", [SANS], 32, 2, T.PILToTensor())
    original = characters.ImageFont.truetype
    calls = 0

    def counted_truetype(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(characters.ImageFont, "truetype", counted_truetype)
    dataset[0]
    dataset[1]
    assert calls == 1
    state = dataset.__getstate__()
    assert state["_font_cache"] == {}
    assert state["_font_cache_pid"] is None


@pytest.mark.parametrize("workers", [0, 1])
def test_dataloader_smoke(workers):
    _, val_transform = characters.build_transforms(_transform_args())
    dataset = characters.SyntheticCharacterDataset("AB", [SANS], 32, 4, val_transform, deterministic=True)
    dataset[0]
    loader = characters.build_loader(
        dataset,
        2,
        workers,
        SequentialSampler(dataset),
        0,
        pin_memory=False,
    )
    images, targets = next(iter(loader))
    assert images.shape == (2, 3, 24, 24)
    assert targets.tolist() == [0, 1]
    del loader
    gc.collect()


def _write_manifest(path, filename, checksum):
    path.write_text(
        json.dumps({
            "version": 1,
            "sources": {"test": {"revision": "abc123"}},
            "fonts": [
                {
                    "family": "DejaVu Sans",
                    "style": "Regular",
                    "source": "test",
                    "filename": filename,
                    "sha256": checksum,
                }
            ],
        }),
        encoding="utf-8",
    )


def test_cli_grid_training_resume_and_provenance(monkeypatch, tmp_path):
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    font_path = font_dir / "Test.ttf"
    shutil.copy2(SANS, font_path)
    manifest = tmp_path / "fonts.json"
    _write_manifest(manifest, font_path.name, characters._sha256(font_path))

    common = [
        "--font-dir",
        str(font_dir),
        "--manifest",
        str(manifest),
        "--alphabet",
        "AB",
        "--arch",
        "convnext_atto",
        "--image-size",
        "16",
        "--render-size",
        "32",
        "--samples-per-epoch",
        "4",
        "--validation-fonts-per-class",
        "1",
        "--batch-size",
        "2",
        "--workers",
        "0",
        "--epochs",
        "1",
        "--device",
        "cpu",
    ]
    grid_path = tmp_path / "grid.png"
    characters.main(characters.get_parser().parse_args([*common, "--show-samples", str(grid_path)]))
    assert grid_path.is_file()

    def tiny_classifier(pretrained=False, *, num_classes=1, **_kwargs):
        assert not pretrained
        return nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, num_classes))

    monkeypatch.setattr(characters.classification, "convnext_atto", tiny_classifier)
    output_dir = tmp_path / "output"
    training_args = [*common, "--output-dir", str(output_dir)]
    characters.main(characters.get_parser().parse_args(training_args))

    checkpoint = output_dir / "checkpoint.pth"
    metadata_path = output_dir / "checkpoint.json"
    assert checkpoint.is_file()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["class_to_index"] == {"A": 0, "B": 1}
    assert metadata["architecture"] == "convnext_atto"
    assert metadata["fonts"][0]["path"] == "Test.ttf"
    assert metadata["manifest"]["source_revisions"] == {"test": "abc123"}
    metadata_before_resume = metadata_path.read_text(encoding="utf-8")
    monkeypatch.setattr(
        characters.ClassificationTrainer,
        "evaluate",
        lambda _self: {"val_loss": float("inf"), "acc1": 0.0, "acc5": 0.0},
    )
    torch_load = torch.load

    def restricted_load(path, **kwargs):
        assert kwargs["weights_only"] is True
        return torch_load(path, **kwargs)

    monkeypatch.setattr(characters.torch, "load", restricted_load)
    characters.main(characters.get_parser().parse_args([*training_args, "--resume", str(checkpoint)]))
    assert metadata_path.read_text(encoding="utf-8") == metadata_before_resume


def test_manifest_checksum_is_verified(tmp_path):
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    font_path = font_dir / "Test.ttf"
    shutil.copy2(SANS, font_path)
    manifest = tmp_path / "fonts.json"
    _write_manifest(manifest, font_path.name, "0" * 64)
    with pytest.raises(ValueError, match="checksum mismatch"):
        characters.resolve_font_records("A", font_dir, manifest)


def test_alexnet_requires_supported_image_size():
    args = characters.get_parser().parse_args(["--arch", "alexnet", "--image-size", "32"])
    with pytest.raises(ValueError, match="at least 63"):
        characters.main(args)


def test_alexnet_dispatch(monkeypatch):
    model = nn.Module()
    kwargs = {}

    def fake_alexnet(**values):
        kwargs.update(values)
        return model

    monkeypatch.setattr(characters, "alexnet", fake_alexnet)

    result = characters.build_model("alexnet", 62)

    assert result is model
    assert kwargs == {"weights": None, "num_classes": 62}
    assert result.avgpool.output_size == 1
    assert result.classifier.in_features == 256
    assert result.classifier.out_features == 62


def test_unmanifested_font_hashes_are_deferred(monkeypatch, tmp_path):
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    shutil.copy2(SANS, font_dir / "Test.ttf")
    hashes = []
    monkeypatch.setattr(characters, "_sha256", lambda path: hashes.append(path) or "hash")

    records, _ = characters.resolve_font_records("A", font_dir, None)
    assert hashes == []
    dataset = characters.SyntheticCharacterDataset("A", records, 32, 1, T.PILToTensor())
    args = characters.get_parser().parse_args(["--font-dir", str(font_dir)])
    metadata = tmp_path / "checkpoint.json"
    characters.write_provenance(metadata, args, dataset, dataset.fonts, None)

    assert hashes == [font_dir / "Test.ttf"]
    assert json.loads(metadata.read_text(encoding="utf-8"))["fonts"][0]["sha256"] == "hash"


def test_fixed_batch_can_lower_loss(monkeypatch, tmp_path):
    torch.manual_seed(0)
    transform = T.Compose([
        T.Resize((8, 8), antialias=True),
        T.Grayscale(num_output_channels=3),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])
    dataset = characters.SyntheticCharacterDataset("AB", [SANS], 32, 2, transform, deterministic=True)
    loader = characters.build_loader(
        dataset,
        2,
        0,
        SequentialSampler(dataset),
        0,
        pin_memory=False,
    )
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))
    criterion = nn.CrossEntropyLoss()
    trainer = characters.ClassificationTrainer(
        model,
        loader,
        loader,
        criterion,
        torch.optim.SGD(model.parameters(), lr=0.2),
        output_file=str(tmp_path / "checkpoint.pth"),
    )
    images, targets = next(iter(loader))
    initial_loss = criterion(model(images), targets).item()
    monkeypatch.setattr(characters.plt, "show", lambda **_kwargs: None)
    trainer.check_setup(lr=0.2, num_it=30)
    final_loss = criterion(model(images), targets).item()
    assert final_loss < initial_loss
