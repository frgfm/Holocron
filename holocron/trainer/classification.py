# Copyright (C) 2019-2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import math
from typing import Any, Dict, Sequence, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm

from .core import Trainer

__all__ = ["BinaryClassificationTrainer", "ClassificationTrainer"]


class ClassificationTrainer(Trainer):
    """Image classification trainer class.

    Args:
        model: model to train
        train_loader: training loader
        val_loader: validation loader
        criterion: loss criterion
        optimizer: parameter optimizer
        gpu: index of the GPU to use
        output_file: path where checkpoints will be saved
        amp: whether to use automatic mixed precision
        skip_nan_loss: whether the optimizer step should be skipped when the loss is NaN
        nan_tolerance: number of consecutive batches with NaN loss before stopping the training
        gradient_acc: number of batches to accumulate the gradient of before performing the update step
        gradient_clip: the gradient clip value
        on_epoch_end: callback triggered at the end of an epoch
    """

    is_binary: bool = False

    @torch.inference_mode()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the validation set

        Returns:
            dict: evaluation metrics
        """
        self.model.eval()

        val_loss, top1, top5, num_samples, num_valid_batches = 0.0, 0, 0, 0, 0
        for x, target in self.val_loader:
            x, target = self.to_cuda(x, target)

            _loss, out = self._get_loss(x, target, return_logits=True)

            # Safeguard for NaN loss
            if not torch.isnan(_loss) and not torch.isinf(_loss):
                val_loss += _loss.item()
                num_valid_batches += 1

            pred = out.topk(5, dim=1)[1] if out.shape[1] >= 5 else out.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            top1 += cast(int, correct[:, 0].sum().item())
            if out.shape[1] >= 5:
                top5 += cast(int, correct.any(dim=1).sum().item())

            num_samples += x.shape[0]

        val_loss /= num_valid_batches

        return {"val_loss": val_loss, "acc1": top1 / num_samples, "acc5": top5 / num_samples}

    @staticmethod
    def _eval_metrics_str(eval_metrics: Dict[str, float]) -> str:
        return (
            f"Validation loss: {eval_metrics['val_loss']:.4} "
            f"(Acc@1: {eval_metrics['acc1']:.2%}, Acc@5: {eval_metrics['acc5']:.2%})"
        )

    @torch.inference_mode()
    def plot_top_losses(
        self,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        classes: Union[Sequence[str], None] = None,
        num_samples: int = 12,
        **kwargs: Any,
    ) -> None:
        # Record loss, prob, target, image
        losses = np.zeros(num_samples, dtype=np.float32)
        preds = np.zeros(num_samples, dtype=int)
        probs = np.zeros(num_samples, dtype=np.float32)
        targets = np.zeros(num_samples, dtype=np.float32 if self.is_binary else int)
        images = [None] * num_samples

        # Switch to unreduced loss
        _reduction = self.criterion.reduction
        self.criterion.reduction = "none"  # type: ignore[assignment]
        self.model.eval()

        train_iter = iter(self.train_loader)

        for x, target in tqdm(train_iter):
            x, target = self.to_cuda(x, target)

            # Forward
            batch_loss, logits = self._get_loss(x, target, return_logits=True)

            # Binary
            if self.is_binary:
                batch_loss = batch_loss.squeeze(1)
                _probs = torch.sigmoid(logits.squeeze(1))
            else:
                _probs = torch.softmax(logits, 1).max(dim=1).values

            if torch.any(batch_loss > losses.min()):
                idcs = np.concatenate((losses, batch_loss.cpu().numpy())).argsort()[-num_samples:]
                kept_idcs = [idx for idx in idcs if idx < num_samples]
                added_idcs = [idx - num_samples for idx in idcs if idx >= num_samples]
                # Update
                losses = np.concatenate((losses[kept_idcs], batch_loss.cpu().numpy()[added_idcs]))
                probs = np.concatenate((probs[kept_idcs], _probs.cpu().numpy()))
                if not self.is_binary:
                    preds = np.concatenate((preds[kept_idcs], logits[added_idcs].argmax(dim=1).cpu().numpy()))
                targets = np.concatenate((targets[kept_idcs], target[added_idcs].cpu().numpy()))
                _imgs = x[added_idcs].cpu() * torch.tensor(std).view(-1, 1, 1)
                _imgs += torch.tensor(mean).view(-1, 1, 1)
                images = [images[idx] for idx in kept_idcs] + [to_pil_image(img) for img in _imgs]

        self.criterion.reduction = _reduction

        if not self.is_binary and classes is None:
            raise AssertionError("arg 'classes' must be specified for multi-class classification")

        # Final sort
        _idcs = losses.argsort()[::-1]
        losses, preds, probs, targets = losses[_idcs], preds[_idcs], probs[_idcs], targets[_idcs]
        images = [images[idx] for idx in _idcs]

        # Plot it
        num_cols = 4
        num_rows = int(math.ceil(num_samples / num_cols))
        _, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5))
        for idx, (img, pred, prob, target, loss) in enumerate(zip(images, preds, probs, targets, losses)):
            _row = int(idx / num_cols)
            _col = idx - num_cols * _row
            axes[_row][_col].imshow(img)
            # Loss, prob, target
            if self.is_binary:
                axes[_row][_col].title.set_text(f"{loss:.3} / {prob:.2} / {target:.2}")
            # Loss, pred (prob), target
            else:
                axes[_row][_col].title.set_text(
                    f"{loss:.3} / {classes[pred]} ({prob:.1%}) / {classes[target]}"  # type: ignore[index]
                )
            axes[_row][_col].axis("off")

        plt.show(**kwargs)


class BinaryClassificationTrainer(ClassificationTrainer):
    """Image binary classification trainer class.

    Args:
        model (torch.nn.Module): model to train
        train_loader (torch.utils.data.DataLoader): training loader
        val_loader (torch.utils.data.DataLoader): validation loader
        criterion (torch.nn.Module): loss criterion
        optimizer (torch.optim.Optimizer): parameter optimizer
        gpu (int, optional): index of the GPU to use
        output_file (str, optional): path where checkpoints will be saved
        amp (bool, optional): whether to use automatic mixed precision
    """

    is_binary: bool = True

    def _get_loss(
        self, x: torch.Tensor, target: torch.Tensor, return_logits: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # In case target are stored as long
        target = target.to(dtype=x.dtype)

        # AMP
        if self.amp:
            with torch.cuda.amp.autocast():
                # Forward
                out = self.model(x)
                # Loss computation
                loss = self.criterion(out, target.view_as(out))
                if return_logits:
                    return loss, out
                return loss

        # Forward
        out = self.model(x)
        loss = self.criterion(out, target.view_as(out))
        if return_logits:
            return loss, out
        return loss

    @torch.inference_mode()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the validation set

        Returns:
            dict: evaluation metrics
        """
        self.model.eval()

        val_loss, top1, num_samples, num_valid_batches = 0.0, 0.0, 0, 0
        for x, target in self.val_loader:
            x, target = self.to_cuda(x, target)

            _loss, out = self._get_loss(x, target, return_logits=True)

            # Safeguard for NaN loss
            if not torch.isnan(_loss) and not torch.isinf(_loss):
                val_loss += _loss.item()
                num_valid_batches += 1

            top1 += torch.sum((target.view_as(out) >= 0.5) == (torch.sigmoid(out) >= 0.5)).item() / out[0].numel()

            num_samples += x.shape[0]

        val_loss /= num_valid_batches

        return {"val_loss": val_loss, "acc": top1 / num_samples}

    @staticmethod
    def _eval_metrics_str(eval_metrics: Dict[str, float]) -> str:
        return f"Validation loss: {eval_metrics['val_loss']:.4} (Acc: {eval_metrics['acc']:.2%})"
