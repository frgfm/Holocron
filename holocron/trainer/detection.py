# Copyright (C) 2019-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torchvision.ops.boxes import box_iou

from .core import Trainer

__all__ = ["DetectionTrainer"]


def assign_iou(gt_boxes: Tensor, pred_boxes: Tensor, iou_threshold: float = 0.5) -> Tuple[List[int], List[int]]:
    """Assigns boxes by IoU"""
    iou = box_iou(gt_boxes, pred_boxes)
    iou = iou.max(dim=1)
    gt_kept = iou.values >= iou_threshold
    assign_unique = torch.unique(iou.indices[gt_kept])
    # Filter
    if iou.indices[gt_kept].shape[0] == assign_unique.shape[0]:
        return torch.arange(gt_boxes.shape[0])[gt_kept], iou.indices[gt_kept]  # type: ignore[return-value]

    gt_indices, pred_indices = [], []
    for pred_idx in assign_unique:
        selection = iou.values[gt_kept][iou.indices[gt_kept] == pred_idx].argmax()
        gt_indices.append(torch.arange(gt_boxes.shape[0])[gt_kept][selection].item())
        pred_indices.append(iou.indices[gt_kept][selection].item())
    return gt_indices, pred_indices  # type: ignore[return-value]


class DetectionTrainer(Trainer):
    """Object detection trainer class

    Args:
        model (torch.nn.Module): model to train
        train_loader (torch.utils.data.DataLoader): training loader
        val_loader (torch.utils.data.DataLoader): validation loader
        criterion (None): loss criterion
        optimizer (torch.optim.Optimizer): parameter optimizer
        gpu (int, optional): index of the GPU to use
        output_file (str, optional): path where checkpoints will be saved
        amp (bool, optional): whether to use automatic mixed precision
        skip_nan_loss (bool, optional): whether the optimizer step should be skipped when the loss is NaN
        on_epoch_end (Callable[[Dict[str, float]], Any]): callback triggered at the end of an epoch
    """

    @staticmethod
    def _to_cuda(  # type: ignore[override]
        x: List[Tensor], target: List[Dict[str, Tensor]]
    ) -> Tuple[List[Tensor], List[Dict[str, Tensor]]]:
        """Move input and target to GPU"""
        x = [_x.cuda(non_blocking=True) for _x in x]
        target = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in target]
        return x, target

    def _backprop_step(self, loss: Tensor, grad_clip: float = 0.1) -> None:
        # Clean gradients
        self.optimizer.zero_grad()
        # Backpropate the loss
        if self.amp:
            self.scaler.scale(loss).backward()
            # Safeguard for Gradient explosion
            if isinstance(grad_clip, float):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)  # type: ignore[attr-defined]

            # Update the params
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            # Safeguard for Gradient explosion
            if isinstance(grad_clip, float):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)  # type: ignore[attr-defined]
            # Update the params
            self.optimizer.step()

    def _get_loss(self, x: List[Tensor], target: List[Dict[str, Tensor]]) -> Tensor:  # type: ignore[override]
        # AMP
        if self.amp:
            with torch.cuda.amp.autocast():  # type: ignore[attr-defined]
                # Forward & loss computation
                loss_dict = self.model(x, target)
                return sum(loss_dict.values())  # type: ignore[return-value]
        # Forward & loss computation
        loss_dict = self.model(x, target)
        return sum(loss_dict.values())  # type: ignore[return-value]

    @staticmethod
    def _eval_metrics_str(eval_metrics: Dict[str, Optional[float]]) -> str:
        loc_str = f"{eval_metrics['loc_err']:.2%}" if isinstance(eval_metrics["loc_err"], float) else "N/A"
        clf_str = f"{eval_metrics['clf_err']:.2%}" if isinstance(eval_metrics["clf_err"], float) else "N/A"
        det_str = f"{eval_metrics['det_err']:.2%}" if isinstance(eval_metrics["det_err"], float) else "N/A"
        return f"Loc error: {loc_str} | Clf error: {clf_str} | Det error: {det_str}"

    @torch.inference_mode()
    def evaluate(self, iou_threshold: float = 0.5) -> Dict[str, Optional[float]]:
        """Evaluate the model on the validation set

        Args:
            iou_threshold (float, optional): IoU threshold for pair assignment

        Returns:
            dict: evaluation metrics
        """
        self.model.eval()

        loc_assigns = 0
        correct, clf_error, loc_fn, loc_fp, num_samples = 0, 0, 0, 0, 0

        for x, target in self.val_loader:
            x, target = self.to_cuda(x, target)

            if self.amp:
                with torch.cuda.amp.autocast():  # type: ignore[attr-defined]
                    detections = self.model(x)
            else:
                detections = self.model(x)

            for dets, t in zip(detections, target):
                if t["boxes"].shape[0] > 0 and dets["boxes"].shape[0] > 0:
                    gt_indices, pred_indices = assign_iou(t["boxes"], dets["boxes"], iou_threshold)
                    loc_assigns += len(gt_indices)
                    _correct = (t["labels"][gt_indices] == dets["labels"][pred_indices]).sum().item()
                else:
                    gt_indices, pred_indices = [], []
                    _correct = 0
                correct += _correct
                clf_error += len(gt_indices) - _correct
                loc_fn += t["boxes"].shape[0] - len(gt_indices)
                loc_fp += dets["boxes"].shape[0] - len(pred_indices)
            num_samples += sum(t["boxes"].shape[0] for t in target)

        nb_preds = num_samples - loc_fn + loc_fp
        # Localization
        loc_err = 1 - 2 * loc_assigns / (nb_preds + num_samples) if nb_preds + num_samples > 0 else None
        # Classification
        clf_err = 1 - correct / loc_assigns if loc_assigns > 0 else None
        # End-to-end
        det_err = 1 - 2 * correct / (nb_preds + num_samples) if nb_preds + num_samples > 0 else None
        return dict(loc_err=loc_err, clf_err=clf_err, det_err=det_err, val_loss=loc_err)
