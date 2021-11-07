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
        assert isinstance(out[0].get('boxes'), torch.Tensor)
        assert isinstance(out[0].get('scores'), torch.Tensor)
        assert isinstance(out[0].get('labels'), torch.Tensor)

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

    # Loss computation with no GT
    gt_boxes = [torch.zeros((0, 4)) for _ in num_boxes]
    gt_labels = [torch.zeros(0, dtype=torch.long) for _ in num_boxes]
    loss = model(x, [dict(boxes=boxes, labels=labels) for boxes, labels in zip(gt_boxes, gt_labels)])


@pytest.mark.parametrize(
    "arch, input_shape",
    [
        ['yolov1', (448, 448)],
        ['yolov2', (416, 416)],
        ['yolov4', (608, 608)],
    ],
)
def test_detection_model(arch, input_shape):
    _test_detection_model(arch, input_shape)
