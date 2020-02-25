import unittest

import math
import torch

from holocron import ops


def _get_boxes():

    return torch.tensor([[0, 0, 100, 100],
                         [50, 50, 100, 100],
                         [50, 50, 150, 150],
                         [100, 100, 200, 200]], dtype=torch.float32)


class Tester(unittest.TestCase):

    def test_iou_penalty(self):

        boxes = _get_boxes()
        penalty = ops.boxes.iou_penalty(boxes, boxes)

        # Check shape
        self.assertEqual(penalty.shape, (boxes.shape[0], boxes.shape[0]))
        # Unit tests
        for idx in range(boxes.shape[0]):
            self.assertEqual(penalty[idx, idx].item(), 0.)

        self.assertEqual(penalty[0, 1].item(), 25 ** 2 / 100 ** 2)
        self.assertEqual(penalty[0, 3].item(), 100 ** 2 / 200 ** 2)
        self.assertEqual(penalty[0, 2].item(), penalty[2, 3].item())

    def test_box_diou(self):

        boxes = _get_boxes()

        diou = ops.boxes.box_diou(boxes, boxes)

        # Check shape
        self.assertEqual(diou.shape, (boxes.shape[0], boxes.shape[0]))
        # Unit tests
        for idx in range(boxes.shape[0]):
            self.assertEqual(diou[idx, idx].item(), 0.)

        self.assertEqual(diou[0, 1].item(), 1 - 0.25 + 25 ** 2 / 100 ** 2)
        self.assertEqual(diou[0, 3].item(), 1 + 100 ** 2 / 200 ** 2)
        self.assertEqual(diou[0, 2].item(), diou[2, 3].item())

    def test_aspect_ratio(self):

        boxes = _get_boxes()
        # All boxes are squares so arctan should yield Pi / 4
        self.assertTrue(torch.equal(ops.boxes.aspect_ratio(boxes), math.pi / 4 * torch.ones(boxes.shape[0])))

    def test_aspect_ratio_consistency(self):

        boxes = _get_boxes()

        # All boxes have the same aspect ratio
        self.assertTrue(torch.equal(ops.boxes.aspect_ratio_consistency(boxes, boxes),
                                    torch.zeros(boxes.shape[0], boxes.shape[0])))

    def test_box_ciou(self):

        boxes = _get_boxes()

        ciou = ops.boxes.box_ciou(boxes, boxes)

        # Check shape
        self.assertEqual(ciou.shape, (boxes.shape[0], boxes.shape[0]))
        # Unit tests
        for idx in range(boxes.shape[0]):
            self.assertEqual(ciou[idx, idx].item(), 0.)
        self.assertEqual(ciou[0, 2].item(), ciou[2, 3].item())


if __name__ == '__main__':
    unittest.main()
