import os
import pytest
import torch

from holocron.models import detection


def _test_detection_model(name, input_size):

    num_classes = 10
    batch_size = 2
    x = torch.rand((batch_size, 3, *input_size))
    model = detection.__dict__[name](pretrained=True, num_classes=num_classes).eval()
    # Check backbone pretrained
    model = detection.__dict__[name](pretrained_backbone=True, num_classes=num_classes).eval()
    with torch.no_grad():
        out = model(x)

    assert isinstance(out, list)
    assert len(out) == x.shape[0]
    if len(out) > 0:
        assert isinstance(out[0].get("boxes"), torch.Tensor)
        assert isinstance(out[0].get("scores"), torch.Tensor)
        assert isinstance(out[0].get("labels"), torch.Tensor)

    # Check that list of Tensors does not change output
    x_list = [torch.rand(3, *input_size) for _ in range(batch_size)]
    with torch.no_grad():
        out_list = model(x_list)
        assert len(out_list) == len(out)

    # Training mode without target
    model = model.train()
    with pytest.raises(ValueError):
        model(x)
    # Generate targets
    num_boxes = [3, 4]
    gt_boxes = []
    for num in num_boxes:
        _boxes = torch.rand((num, 4), dtype=torch.float)
        # Ensure format xmin, ymin, xmax, ymax
        _boxes[:, :2] *= _boxes[:, 2:]
        # Ensure some anchors will be assigned
        _boxes[0, :2] = 0
        _boxes[0, 2:] = 1
        # Check cases where cell can get two assignments
        _boxes[1, :2] = 0.2
        _boxes[1, 2:] = 0.8
        gt_boxes.append(_boxes)
    gt_labels = [(num_classes * torch.rand(num)).to(dtype=torch.long) for num in num_boxes]

    # Loss computation
    loss = model(x, [dict(boxes=boxes, labels=labels) for boxes, labels in zip(gt_boxes, gt_labels)])
    assert isinstance(loss, dict)
    for subloss in loss.values():
        assert isinstance(subloss, torch.Tensor)
        assert subloss.requires_grad
        assert not torch.isnan(subloss)

    # Loss computation with no GT
    gt_boxes = [torch.zeros((0, 4)) for _ in num_boxes]
    gt_labels = [torch.zeros(0, dtype=torch.long) for _ in num_boxes]
    loss = model(x, [dict(boxes=boxes, labels=labels) for boxes, labels in zip(gt_boxes, gt_labels)])
    sum(v for v in loss.values()).backward()


@pytest.mark.parametrize(
    "arch, input_shape",
    [
        ["yolov1", (448, 448)],
        ["yolov2", (416, 416)],
        ["yolov4", (608, 608)],
    ],
)
def test_detection_model(arch, input_shape):
    _test_detection_model(arch, input_shape)


@pytest.mark.parametrize(
    "arch, input_shape",
    [
        ["yolov1", (448, 448)],
        ["yolov2", (416, 416)],
        ["yolov4", (608, 608)],
    ],
)
def test_detection_onnx_export(arch, input_shape, tmpdir_factory):
    model = detection.__dict__[arch](pretrained=False, num_classes=10).eval()
    tmp_path = os.path.join(str(tmpdir_factory.mktemp("onnx")), f"{arch}.onnx")
    img_tensor = torch.rand((1, 3, *input_shape))
    with torch.no_grad():
        torch.onnx.export(model, img_tensor, tmp_path, export_params=True, opset_version=14)


@torch.inference_mode()
def test_yolov1():
    input_shape = (448, 448)
    n, h, w = 2, 7, 7
    num_anchors = 2
    num_classes = 10
    model = detection.yolov1(num_classes=10, pretrained_backbone=False)

    # Forward
    t = torch.rand((n, 3, *input_shape), dtype=torch.float32)
    out = model._forward(t)
    assert out.shape == (n, h * w * (num_anchors * 5 + num_classes))

    # Format outputs
    t = torch.rand((n, h * w * (num_anchors * 5 + num_classes)), dtype=torch.float32)
    b_coords, b_o, b_scores = model._format_outputs(t)
    assert b_coords.shape == (n, h, w, num_anchors, 4)
    assert b_o.shape == (n, h, w, num_anchors)
    assert b_scores.shape == (n, h, w, 1, num_classes)
    assert torch.all(b_coords <= 1) and torch.all(b_coords >= 0)
    assert torch.all(b_o <= 1) and torch.all(b_o >= 0)
    assert torch.allclose(b_scores.sum(-1), torch.ones(1))

    # Compute loss
    target = [
        dict(
            boxes=torch.tensor([[0, 0, 1 / 7, 1 / 7]], dtype=torch.float32), labels=torch.zeros((1,), dtype=torch.long)
        )
    ]
    pred_boxes = torch.zeros((1, h, w, num_anchors, 4), dtype=torch.float32)
    pred_boxes[..., :2] = 0.5
    pred_boxes[..., 2:] = 1 / 7
    pred_boxes[0, 0, 0, 1, 0] = 0.8
    pred_o = torch.zeros((1, h, w, num_anchors), dtype=torch.float32)
    pred_o[0, 0, 0, 0] = 0.5
    pred_o[0, -1, -1, 0] = 0.5
    pred_scores = torch.zeros((1, h, w, 1, num_classes), dtype=torch.float32)
    pred_scores[0, 0, 0, 0, 0] = 0.5
    pred_scores[0, 0, 0, 0, 1:] = 0.5 / (num_classes - 1)
    loss_dict = model._compute_losses(pred_boxes, pred_o, pred_scores, target, ignore_high_iou=True)
    assert loss_dict["obj_loss"].item() == model.lambda_obj * 0.5**2
    assert loss_dict["noobj_loss"].item() == model.lambda_noobj * 0.5**2
    assert loss_dict["bbox_loss"].item() == 0
    assert (
        abs(
            loss_dict["clf_loss"].item()
            - model.lambda_class * (0.5**2 + (num_classes - 1) * (0.5 / (num_classes - 1)) ** 2)
        )
        < 1e-7
    )

    # Post process
    b_coords = torch.zeros((n, h * w * num_anchors, 4), dtype=torch.float32)
    b_coords[..., :2] = 0.5
    b_coords[..., 2:] = 1 / h
    b_o = torch.zeros((n, h * w * num_anchors), dtype=torch.float32)
    b_o[:, ::2] = 0.5
    b_scores = torch.zeros((n, h * w * num_anchors, num_classes), dtype=torch.float32)
    b_scores[..., 0] = 0.5
    b_scores[..., 1:] = 0.5 / (num_classes - 1)
    dets = model.post_process(b_coords, b_o, b_scores, (h, w))
    assert dets[0]["labels"].shape[0] == b_o.shape[1] // 2
    assert torch.all(dets[0]["labels"] == 0)
    assert torch.all(dets[0]["scores"] == 0.25)
    assert torch.equal(dets[0]["boxes"][0], torch.tensor([0, 0, 1 / 7, 1 / 7]))
    assert torch.allclose(dets[0]["boxes"][-1], torch.tensor([6 / 7, 6 / 7, 1, 1]))


@torch.inference_mode()
def test_yolov2():
    input_shape = (416, 416)
    n, h, w = 2, 13, 13
    num_anchors = 5
    num_classes = 10
    model = detection.yolov2(num_classes=10, pretrained_backbone=False)

    # Forward
    t = torch.rand((n, 3, *input_shape), dtype=torch.float32)
    out = model._forward(t)
    assert out.shape == (n, num_anchors * (5 + num_classes), h, w)

    # Format outputs
    t = torch.rand((n, num_anchors * (5 + num_classes), h, w), dtype=torch.float32)
    b_coords, b_o, b_scores = model._format_outputs(t)
    assert b_coords.shape == (n, h, w, num_anchors, 4)
    assert b_o.shape == (n, h, w, num_anchors)
    assert b_scores.shape == (n, h, w, num_anchors, num_classes)
    assert torch.all(b_coords[..., :2] <= 1) and torch.all(b_coords >= 0)
    assert torch.all(b_o <= 1) and torch.all(b_o >= 0)
    assert torch.allclose(b_scores.sum(-1), torch.ones(1))

    # Compute loss
    target = [dict(boxes=torch.tensor([[0, 0, 1, 1]], dtype=torch.float32), labels=torch.zeros((1,), dtype=torch.long))]
    pred_boxes = torch.zeros((1, h, w, num_anchors, 4), dtype=torch.float32)
    pred_boxes[..., :2] = 0.5
    pred_boxes[..., 2:] = 1
    pred_boxes[0, -1, -1, 0, 0] = (w - 1) / w
    pred_boxes[0, -1, -1, 0, 1] = (h - 1) / h
    pred_boxes[0, -1, -1, 0, 2] = 1 / w
    pred_boxes[0, -1, -1, 0, 3] = 1 / h
    pred_o = torch.zeros((1, h, w, num_anchors), dtype=torch.float32)
    pred_o[0, h // 2, w // 2, 0] = 0.5
    pred_o[0, -1, -1, 0] = 0.5
    pred_scores = torch.zeros((1, h, w, 1, num_classes), dtype=torch.float32)
    pred_scores[0, h // 2, w // 2, 0, 0] = 0.5
    pred_scores[0, h // 2, w // 2, 0, 1:] = 0.5 / (num_classes - 1)
    loss_dict = model._compute_losses(pred_boxes, pred_o, pred_scores, target, ignore_high_iou=True)
    assert loss_dict["obj_loss"].item() == model.lambda_obj * 0.5**2
    assert loss_dict["noobj_loss"].item() == model.lambda_noobj * 0.5**2
    assert loss_dict["bbox_loss"].item() == 0
    assert (
        abs(
            loss_dict["clf_loss"].item()
            - model.lambda_class * (0.5**2 + (num_classes - 1) * (0.5 / (num_classes - 1)) ** 2)
        )
        < 1e-7
    )

    # Post process
    b_coords = torch.zeros((n, h * w * num_anchors, 4), dtype=torch.float32)
    b_coords[..., :2] = 0.5
    b_coords[..., 2:] = 1
    b_o = torch.zeros((n, h * w * num_anchors), dtype=torch.float32)
    b_o[:, ::2] = 0.5
    b_scores = torch.zeros((n, h * w * num_anchors, num_classes), dtype=torch.float32)
    b_scores[..., 0] = 0.5
    b_scores[..., 1:] = 0.5 / (num_classes - 1)
    dets = model.post_process(b_coords, b_o, b_scores, (h, w))
    assert dets[0]["labels"].shape[0] == 1
    assert torch.all(dets[0]["labels"] == 0)
    assert torch.all(dets[0]["scores"] == 0.25)
    assert torch.equal(dets[0]["boxes"][0], torch.tensor([0, 0, 1, 1], dtype=torch.float32))
