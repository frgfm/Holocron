import math
import re
from doctest import DocTestParser, DocTestRunner
from pathlib import Path
from xml.etree import ElementTree as ET  # noqa: S405

import pytest
import torch
from PIL import Image
from torchvision.transforms import v2 as T

from holocron.models.classification import ReXNet1_0x_Checkpoint
from holocron.nn import PolyLoss


@pytest.mark.parametrize("document", ["README.md", "docs/docs/index.md"])
def test_homepage_quickstart(monkeypatch, tmp_path, document):
    homepage = (Path(__file__).parents[1] / document).read_text()
    match = re.search(
        r"<!-- quickstart-example-start -->\s*```python\n(?P<code>.*?)\n```\s*<!-- quickstart-example-end -->",
        homepage,
        re.DOTALL,
    )
    assert match is not None

    image_path = tmp_path / "image.png"
    # Non-square and non-uniform, so stretching or missing pixel scaling changes the tensor
    Image.linear_gradient("L").resize((320, 240)).convert("RGB").save(image_path)

    monkeypatch.setattr("holocron.models.utils.load_pretrained_params", lambda *_args, **_kwargs: None)
    namespace = {"path_to_an_image": image_path}
    exec(compile(match["code"], document, "exec"), namespace)  # noqa: S102

    checkpoint = namespace["checkpoint"]
    probabilities = namespace["probabilities"]
    pre = checkpoint.pre_processing
    expected = T.Compose([
        T.Resize(232, interpolation=pre.interpolation),
        T.CenterCrop(224),
        T.PILToTensor(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(pre.mean, pre.std),
    ])
    image = Image.open(image_path).convert("RGB")
    assert torch.equal(namespace["input_tensor"], expected(image).unsqueeze(0))
    assert checkpoint.transforms(train=True)(image).shape == pre.input_shape
    # ImageNet-1K ReXNet weights are timm ports: bicubic 256 resize, then a 224 center crop
    imagenet = ReXNet1_0x_Checkpoint.IMAGENET1K.value.pre_processing
    timm_eval = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.PILToTensor(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(imagenet.mean, imagenet.std),
    ])
    assert torch.equal(ReXNet1_0x_Checkpoint.IMAGENET1K.value.transforms()(image), timm_eval(image))
    assert probabilities.shape == (len(checkpoint.meta.categories),)
    assert namespace["label"] in checkpoint.meta.categories
    assert 0 <= namespace["confidence"] <= 1


def test_poly_loss_example():
    example = DocTestParser().get_doctest(PolyLoss.__doc__, {}, "PolyLoss", "loss.py", 0)
    result = DocTestRunner().run(example)
    assert result.failed == 0
    assert result.attempted > 0


def test_checkpoint_chart_matches_table():
    repo_root = Path(__file__).parents[1]
    models_page = (repo_root / "docs" / "docs" / "reference" / "models" / "models.md").read_text()
    documented = {
        match["checkpoint"]: (float(match["acc1"]), float(match["params"]))
        for match in re.finditer(
            r"^\| \[`(?P<checkpoint>[^`]+\.IMAGENETTE)`\]\[[^\]]+\] \| "
            r"(?P<acc1>\d+\.\d+)% \| [^|]+ \| (?P<params>\d+(?:\.\d+)?)M \|",
            models_page,
            re.MULTILINE,
        )
    }

    # The SVG is a trusted repository asset, not user input.
    chart = ET.parse(repo_root / "docs" / "docs" / "img" / "checkpoint-accuracy-vs-parameters.svg")  # noqa: S314
    svg = "{http://www.w3.org/2000/svg}"
    points = chart.findall(f".//{svg}g[@data-checkpoint]")
    plotted = {
        point.attrib["data-checkpoint"]: (float(point.attrib["data-acc1"]), float(point.attrib["data-params"]))
        for point in points
    }

    assert len(points) == len(plotted) == len(documented) == 27
    assert plotted == documented
    assert [point.attrib["data-checkpoint"] for point in points if point.get("data-default") == "true"] == [
        "ResNet18_Checkpoint.IMAGENETTE"
    ]

    positions = {}
    for point in points:
        circle = point.find(f"{svg}circle")
        if circle is not None:
            x, y = float(circle.attrib["cx"]), float(circle.attrib["cy"])
        else:
            diamond = point.find(f"{svg}path")
            assert diamond is not None
            coordinates = [float(value) for value in re.findall(r"[\d.]+", diamond.attrib["d"])]
            x = sum(coordinates[::2]) / (len(coordinates) / 2)
            y = sum(coordinates[1::2]) / (len(coordinates) / 2)

        positions[point.attrib["data-checkpoint"]] = (x, y)
        acc1, params = plotted[point.attrib["data-checkpoint"]]
        expected_x = 90 + (math.log10(params) - math.log10(3)) / (math.log10(200) - math.log10(3)) * 750
        expected_y = 570 - (acc1 - 87) / (96 - 87) * 470
        assert math.isclose(x, expected_x, abs_tol=0.11)
        assert math.isclose(y, expected_y, abs_tol=0.11)

    expected_pareto = {
        checkpoint
        for checkpoint, (acc1, params) in documented.items()
        if not any(
            other_checkpoint != checkpoint
            and other_params <= params
            and other_acc1 >= acc1
            and (other_params < params or other_acc1 > acc1)
            for other_checkpoint, (other_acc1, other_params) in documented.items()
        )
    }
    plotted_pareto = {point.attrib["data-checkpoint"] for point in points if point.get("data-pareto") == "true"}
    assert plotted_pareto == expected_pareto

    frontier = chart.find(f".//{svg}polyline[@class='frontier-line']")
    assert frontier is not None
    coordinates = [float(value) for value in re.findall(r"[\d.]+", frontier.attrib["points"])]
    frontier_points = list(zip(coordinates[::2], coordinates[1::2], strict=True))
    expected_frontier = [
        positions[checkpoint] for checkpoint in sorted(expected_pareto, key=lambda name: documented[name][1])
    ]
    assert frontier_points == expected_frontier
