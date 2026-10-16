import copy

import pytest
import torch
from torch.utils.data import DataLoader

from holocron.models.detection import yolo26n
from holocron.models.detection.yolo26 import _assign, _paired_ciou  # noqa: PLC2701
from holocron.trainer import DetectionTrainer


def _targets():
    return [
        {"boxes": torch.tensor([[0.1, 0.2, 0.6, 0.8], [0.65, 0.4, 0.9, 0.7]]), "labels": torch.tensor([0, 1])},
        {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)},
    ]


@pytest.mark.parametrize("use_amp", [False, True])
def test_yolo26_training_ragged_targets_and_empty_images(use_amp):
    torch.manual_seed(1)
    model = yolo26n(num_classes=2)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=use_amp):
        losses = model(torch.rand(2, 3, 64, 96), _targets())
    assert len(losses) == 4
    assert all(value.isfinite() and value.requires_grad for value in losses.values())
    sum(losses.values()).backward()
    assert all(parameter.grad is not None and parameter.grad.isfinite().all() for parameter in model.parameters())
    model.zero_grad(set_to_none=True)
    empty = [_targets()[1], _targets()[1]]
    losses = model(torch.rand(2, 3, 64, 64), empty)
    assert losses["one_box_loss"] == losses["many_box_loss"] == 0
    sum(losses.values()).backward()
    assert model.one_to_one.classes[0][-1].bias.grad.abs().sum() > 0


@pytest.mark.parametrize("box_dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_yolo26_training_accepts_floating_box_dtypes(box_dtype):
    torch.manual_seed(5)
    model = yolo26n(num_classes=2)
    images = torch.rand(2, 3, 64, 64)
    targets = [{**target, "boxes": target["boxes"].to(box_dtype)} for target in _targets()]
    original_targets = copy.deepcopy(targets)
    float_targets = [{**target, "boxes": target["boxes"].float()} for target in targets]
    with torch.no_grad():
        reference_losses = copy.deepcopy(model)(images, float_targets)
    losses = model(images, targets)
    torch.testing.assert_close(losses, reference_losses)
    assert all(value.isfinite() for value in losses.values())
    sum(losses.values()).backward()
    assert all(parameter.grad is not None and parameter.grad.isfinite().all() for parameter in model.parameters())
    torch.testing.assert_close(targets, original_targets)


def test_yolo26_one_to_one_branch_detaches_backbone():
    model = yolo26n(num_classes=2)
    losses = model(torch.rand(2, 3, 64, 64), _targets())
    (losses["one_box_loss"] + losses["one_class_loss"]).backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert model.one_to_one.boxes[0][-1].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.one_to_many.parameters())


def test_yolo26_assignment_matches_labels_and_small_objects():
    boxes = torch.tensor([[0.1, 0.1, 0.4, 0.4], [0.15, 0.1, 0.4, 0.4], [0.7, 0.7, 0.9, 0.9]])
    points = torch.tensor([[0.25, 0.25], [0.3, 0.25], [0.75, 0.75]])
    target = {"boxes": boxes[[0, 2]], "labels": torch.tensor([1, 0])}
    logits = torch.tensor([[-2.0, 4.0], [-2.0, 3.0], [3.0, -2.0]])
    matched, quality, positive = _assign(boxes, logits, points, target, 1)
    assert positive.tolist() == [True, False, True]
    assert quality[0, 1] == quality[2, 0] == 1
    assert torch.equal(matched[positive], target["boxes"])
    tiny = {"boxes": torch.tensor([[0.255, 0.255, 0.26, 0.26]]), "labels": torch.tensor([0])}
    assert _assign(boxes, logits, points, tiny, 1)[2].sum() == 1
    # Even a prediction with zero IoU must receive a box regression signal.
    wrong_boxes = boxes + 2
    assert _assign(wrong_boxes, logits, points, tiny, 1)[1].sum() > 0


def test_yolo26_ciou_is_differentiable_and_zero_for_identical_boxes():
    target = torch.tensor([[0.1, 0.1, 0.7, 0.9]])
    assert _paired_ciou(target, target).abs().max() < 1e-6
    prediction = torch.tensor([[0.2, 0.3, 0.6, 0.8]], requires_grad=True)
    _paired_ciou(prediction, target).sum().backward()
    assert prediction.grad.isfinite().all()
    assert (prediction.grad != 0).all()


def test_yolo26_nms_free_and_optional_nms():
    model = yolo26n(num_classes=2, box_score_thresh=0.5, max_detections=2).eval()
    boxes = torch.tensor([[[0.1, 0.1, 0.8, 0.8], [0.1, 0.1, 0.8, 0.8], [0.0, 0.0, 0.1, 0.1]]])
    logits = torch.tensor([[[4.0, -4.0], [3.0, -4.0], [-4.0, -4.0]]])
    # NMS-free means no hidden duplicate-removal step; training learns uniqueness.
    assert len(model.post_process(boxes, logits)[0]["boxes"]) == 2
    model.nms = True
    assert len(model.post_process(boxes, logits)[0]["boxes"]) == 1
    assert len(model.post_process(boxes + 2, logits)[0]["boxes"]) == 0
    # Saturated scores must retain the class with the largest logit.
    assert model.post_process(boxes[:, :1], logits.new_tensor([[[20, 30]]]))[0]["labels"].item() == 1


def test_yolo26_detection_trainer_tuple_collation():
    model = yolo26n(num_classes=2)
    samples = list(zip(torch.rand(2, 3, 64, 64), _targets(), strict=True))
    loader = DataLoader(samples, batch_size=2, collate_fn=lambda batch: tuple(zip(*batch, strict=True)))
    trainer = DetectionTrainer(model, loader, loader, None, torch.optim.SGD(model.parameters(), lr=1e-3))
    images, targets = next(iter(loader))
    loss = trainer._get_loss(images, targets)
    loss.backward()
    assert loss.isfinite()
    assert trainer.evaluate()["det_err"] is not None


@pytest.mark.parametrize("nms", [False, True])
def test_yolo26_deployment_parity_and_serialization(nms, tmp_path):
    torch.manual_seed(3)
    model = yolo26n(num_classes=2, box_score_thresh=0, nms=nms).eval()
    images = torch.rand(2, 3, 64, 96)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.uniform_(-0.2, 0.2)
                module.running_var.uniform_(0.5, 1.5)
                module.weight.uniform_(0.8, 1.2)
                module.bias.uniform_(-0.1, 0.1)
    with torch.inference_mode():
        before = model(images)
        model_from_list = model(list(images.unbind()))
        deployed = model.to_deploy()
        after = deployed(images)
        # Verify each feature point before top-k can reorder nearly tied scores.
        features = model.neck(model.backbone(images))
        raw_before = model._inference_head()(features)
        raw_after = deployed._inference_head()(deployed.neck(deployed.backbone(images)))
        torch.testing.assert_close(raw_before, raw_after, atol=1e-6, rtol=1e-5)
        if not nms:
            dynamic_shapes = (tuple({2: torch.export.Dim.DYNAMIC, 3: torch.export.Dim.DYNAMIC} for _ in features),)
            exported = torch.export.export(model._inference_head(), (features,), dynamic_shapes=dynamic_shapes)
            features = tuple(feature.transpose(-2, -1) for feature in features)
            torch.testing.assert_close(exported.module()(features), model._inference_head()(features))
    assert not model.deployed
    assert deployed.deployed
    assert sum(parameter.numel() for parameter in deployed.parameters()) < sum(
        parameter.numel() for parameter in model.parameters()
    )
    for reference, candidate, listed in zip(before, after, model_from_list, strict=True):
        # Fusion can change the order of tied class probabilities. Match the
        # returned boxes one-to-one, then check their coordinates/classes/scores.
        pairing = torch.cdist(reference["boxes"], candidate["boxes"], p=1).argmin(1)
        assert pairing.unique().numel() == len(reference["boxes"]) == len(candidate["boxes"])
        for name in reference:
            torch.testing.assert_close(reference[name], candidate[name][pairing], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(reference[name], listed[name])
    path = tmp_path / "detector.pt"
    torch.save(deployed.state_dict(), path)
    restored = yolo26n(num_classes=2, box_score_thresh=0, nms=nms).eval().to_deploy()
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.inference_mode():
        torch.testing.assert_close(restored(images), after)
    with pytest.raises(RuntimeError, match="cannot train"):
        deployed.train()(images, _targets())
    deployed.eval().nms = not nms
    with pytest.raises(RuntimeError, match="selected head was removed"):
        deployed(images)
    with pytest.raises(RuntimeError, match="selected head was removed"):
        deployed.to_deploy()


def test_yolo26_parameter_budget():
    model = yolo26n().eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_572_280
    assert sum(parameter.numel() for parameter in model.to_deploy().parameters()) == 2_408_932


def test_yolo26_rejects_unavailable_weights_and_invalid_inputs():
    with pytest.raises(ValueError, match="No Holocron"):
        yolo26n(pretrained=True)
    with pytest.raises(ValueError, match="No Holocron"):
        yolo26n(pretrained_backbone=True)
    model = yolo26n(num_classes=2)
    with pytest.raises(ValueError, match="Training requires"):
        model(torch.rand(2, 3, 64, 64))
    with pytest.raises(ValueError, match="divisible"):
        model(torch.rand(2, 3, 65, 64), _targets())
    with pytest.raises(ValueError, match="equally sized"):
        model([torch.rand(3, 64, 64), torch.rand(3, 96, 64)], _targets())
    invalid = copy.deepcopy(_targets())
    invalid[0]["boxes"][0, 2] = 2
    with pytest.raises(ValueError, match="normalized"):
        model(torch.rand(2, 3, 64, 64), invalid)
    with pytest.raises(ValueError, match="eval"):
        model.to_deploy()


def test_yolo26_postprocess_preserves_saturated_class_order():
    model = yolo26n(num_classes=3, box_score_thresh=0.0, max_detections=1)
    boxes = torch.tensor([[[0.1, 0.1, 0.9, 0.9]]])
    logits = torch.tensor([[[-25.0, -24.0, -23.0]]], dtype=torch.float16)
    assert not logits.sigmoid().count_nonzero()
    prediction = model.post_process(boxes, logits)[0]
    assert prediction["labels"].tolist() == [2]
