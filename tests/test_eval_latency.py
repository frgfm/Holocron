import json
import os
import subprocess  # noqa: S404
import sys
from datetime import datetime
from pathlib import Path

import pytest
import torch

from holocron import models
from scripts import eval_latency

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("arch", "task", "shapes"),
    [
        ("repvit_m0_9", "classification", [[1, 3, 64, 64]]),
        ("yolo26n", "detection", [[1, 3, 64, 64]]),
        ("unet_tvresnet34", "segmentation", [[1, 3, 64, 64]]),
        ("CharacterClassifier", "recognition", [[1, 1, 32, 64], [1]]),
        ("CTCRecognizer", "recognition", [[1, 1, 32, 64], [1]]),
    ],
)
def test_inference_cli_report(arch, task, shapes, tmp_path):
    path = tmp_path / "records" / "inference.json"
    command = [sys.executable, str(ROOT / "scripts/eval_latency.py"), arch]
    command += ["--size", "64", "--num-classes", "3", "--it", "2", "--warmup", "1", "--repeat", "2", "--output"]
    subprocess.run(  # noqa: S603
        [*command, str(path)], cwd=ROOT, capture_output=True, text=True, check=True
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    assert datetime.fromisoformat(report["started_at"]).utcoffset().total_seconds() == 0
    assert report["config"]["arch"] == arch
    assert report["model"]["task"] == task
    assert report["model"]["input_shapes"] == shapes
    assert report["model"]["tested_path"].endswith(".forward")
    assert (ROOT / report["model"]["source_path"]).is_file()
    assert report["model"]["reparametrized"] == (task == "classification")
    assert report["environment"]["processor"]
    assert report["environment"]["versions"]["torch"]
    assert len(report["environment"]["revision"]) == 40
    runs = report["runs"]
    assert len({run["pid"] for run in runs}) == 2
    assert all(run["pid"] != os.getpid() for run in runs)
    assert all(run["runtime"]["torch_version"] == torch.__version__ for run in runs)
    assert report["summary"]["median_ms"] > 0
    if sys.platform != "win32":
        assert report["summary"]["peak_rss_mib"] == max(run["peak_rss_mib"] for run in runs)
        assert report["summary"]["peak_rss_mib"] > 0


def test_model_discovery_includes_vision_and_recognition():
    assert set(eval_latency.list_models()) == {*models.list_models(), "CharacterClassifier", "CTCRecognizer"}


@pytest.mark.parametrize("failed_index", [0, 1])
def test_partial_suite_is_marked_incomplete(monkeypatch, tmp_path, failed_index):
    names = ["resnet18", "CTCRecognizer"]
    monkeypatch.setattr(eval_latency, "run_child", lambda *_args: json.dumps(names))

    def benchmark(args):
        if args.arch == names[failed_index]:
            raise RuntimeError("inference failed")
        return {"config": {"arch": args.arch}}

    monkeypatch.setattr(eval_latency, "benchmark", benchmark)
    output = tmp_path / "partial.json"
    output.write_text('{"complete": true}', encoding="utf-8")
    args = eval_latency.get_parser().parse_args(["all", "--output", str(output)])
    with pytest.raises(RuntimeError, match="inference failed"):
        eval_latency.main(args)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["expected_models"] == names
    assert not report["complete"]
    assert [row["config"]["arch"] for row in report["benchmarks"]] == names[:failed_index]


def test_nonfinite_detection_output_is_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        eval_latency.check_finite([{"boxes": torch.zeros(0, 4), "scores": torch.tensor([float("nan")])}])
