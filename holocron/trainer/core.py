import math
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, MultiplicativeLR  # type: ignore[attr-defined]
from torchvision.ops.boxes import box_iou
from fastprogress.fastprogress import ConsoleMasterBar
from fastprogress import master_bar, progress_bar
from typing import Optional, Dict, Any, Union, List, Tuple

from contiguous_params import ContiguousParams

from .utils import freeze_bn, freeze_model


__all__ = ['Trainer', 'ClassificationTrainer', 'SegmentationTrainer', 'DetectionTrainer']


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        gpu: Optional[int] = None,
        output_file: str = './checkpoint.pth'
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer

        # Output file
        self.output_file = output_file

        # Initialize
        self.step = 0
        self.start_epoch = 0
        self.epoch = 0
        self.min_loss = math.inf
        self.gpu = gpu
        self._params: Optional[ContiguousParams] = None
        self.lr_recorder: List[float] = []
        self.loss_recorder: List[float] = []
        self.set_device(gpu)
        self._reset_opt(self.optimizer.defaults['lr'])

    def set_device(self, gpu: Optional[int] = None) -> None:
        """Move tensor objects to the target GPU

        Args:
            gpu: index of the target GPU device
        """
        if isinstance(gpu, int):
            if not torch.cuda.is_available():
                raise AssertionError("PyTorch cannot access your GPU. Please investigate!")
            if gpu >= torch.cuda.device_count():
                raise ValueError("Invalid device index")
            torch.cuda.set_device(gpu)
            self.model = self.model.cuda()
            if isinstance(self.criterion, torch.nn.Module):
                self.criterion = self.criterion.cuda()

    def save(self, output_file: str) -> None:
        """Save a trainer checkpoint

        Args:
            output_file: destination file path
        """
        torch.save(dict(epoch=self.epoch, step=self.step, min_loss=self.min_loss,
                        optimizer=self.optimizer.state_dict(),
                        model=self.model.state_dict()),
                   output_file,
                   _use_new_zipfile_serialization=False)

    def load(self, state: Dict[str, Any]) -> None:
        """Resume from a trainer state

        Args:
            state (dict): checkpoint dictionary
        """
        self.start_epoch = state['epoch']
        self.epoch = self.start_epoch
        self.step = state['step']
        self.min_loss = state['min_loss']
        self.optimizer.load_state_dict(state['optimizer'])
        self.model.load_state_dict(state['model'])

    def _fit_epoch(self, mb: ConsoleMasterBar) -> None:
        """Fit a single epoch

        Args:
            mb (fastprogress.master_bar): primary progress bar
        """
        self.model = freeze_bn(self.model.train())

        pb = progress_bar(self.train_loader, parent=mb)
        for x, target in pb:
            x, target = self.to_cuda(x, target)

            # Forward
            batch_loss = self._get_loss(x, target)

            # Backprop
            self._backprop_step(batch_loss)
            # Update LR
            self.scheduler.step()
            pb.comment = f"Training loss: {batch_loss.item():.4}"

            self.step += 1
        self.epoch += 1

    def to_cuda(
        self,
        x: Tensor,
        target: Union[Tensor, List[Dict[str, Tensor]]]
    ) -> Tuple[Tensor, Union[Tensor, List[Dict[str, Tensor]]]]:
        """Move input and target to GPU"""
        if isinstance(self.gpu, int):
            if self.gpu >= torch.cuda.device_count():
                raise ValueError("Invalid device index")
            return self._to_cuda(x, target)  # type: ignore[arg-type]
        else:
            return x, target

    @staticmethod
    def _to_cuda(x: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """Move input and target to GPU"""
        x = x.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        return x, target

    def _backprop_step(self, loss: Tensor) -> None:
        # Clean gradients
        self.optimizer.zero_grad()
        # Backpropate the loss
        loss.backward()
        # Update the params
        self.optimizer.step()

    def _get_loss(self, x: Tensor, target: Tensor) -> Tensor:
        # Forward
        out = self.model(x)
        # Loss computation
        return self.criterion(out, target)

    def _set_params(self) -> None:
        self._params = ContiguousParams([p for p in self.model.parameters() if p.requires_grad])

    def _reset_opt(self, lr: float) -> None:
        """Reset the target params of the optimizer"""
        self.optimizer.defaults['lr'] = lr
        self.optimizer.state = defaultdict(dict)
        self.optimizer.param_groups = []
        self._set_params()
        self.optimizer.add_param_group(dict(params=self._params.contiguous()))  # type: ignore[union-attr]

    @torch.no_grad()
    def evaluate(self):
        raise NotImplementedError

    @staticmethod
    def _eval_metrics_str(eval_metrics):
        raise NotImplementedError

    def _reset_scheduler(self, lr: float, num_epochs: int, sched_type: str = 'onecycle') -> None:
        if sched_type == 'onecycle':
            self.scheduler = OneCycleLR(self.optimizer, lr, num_epochs * len(self.train_loader))
        elif sched_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, num_epochs * len(self.train_loader), eta_min=lr / 25e4)
        else:
            raise ValueError(f"The following scheduler type is not supported: {sched_type}")

    def fit_n_epochs(
        self,
        num_epochs: int,
        lr: float,
        freeze_until: Optional[str] = None,
        sched_type: str = 'onecycle'
    ) -> None:
        """Train the model for a given number of epochs

        Args:
            num_epochs (int): number of epochs to train
            lr (float): learning rate to be used by the scheduler
            freeze_until (str, optional): last layer to freeze
            sched_type (str, optional): type of scheduler to use
        """

        self.model = freeze_model(self.model.train(), freeze_until)
        # Update param groups & LR
        self._reset_opt(lr)
        # Scheduler
        self._reset_scheduler(lr, num_epochs, sched_type)

        mb = master_bar(range(num_epochs))
        for _ in mb:

            self._fit_epoch(mb)
            # Check whether ops invalidated the buffer
            self._params.assert_buffer_is_valid()  # type: ignore[union-attr]
            eval_metrics = self.evaluate()

            # master bar
            mb.main_bar.comment = f"Epoch {self.start_epoch + self.epoch}/{self.start_epoch + num_epochs}"
            mb.write(f"Epoch {self.start_epoch + self.epoch}/{self.start_epoch + num_epochs} - "
                     f"{self._eval_metrics_str(eval_metrics)}")

            if eval_metrics['val_loss'] < self.min_loss:
                print(f"Validation loss decreased {self.min_loss:.4} --> "
                      f"{eval_metrics['val_loss']:.4}: saving state...")
                self.min_loss = eval_metrics['val_loss']
                self.save(self.output_file)

    def lr_find(
        self,
        freeze_until: Optional[str] = None,
        start_lr: float = 1e-7,
        end_lr: float = 1,
        num_it: int = 100
    ) -> None:
        """Gridsearch the optimal learning rate for the training

        Args:
           freeze_until (str, optional): last layer to freeze
           start_lr (float, optional): initial learning rate
           end_lr (float, optional): final learning rate
           num_it (int, optional): number of iterations to perform
        """

        self.model = freeze_model(self.model.train(), freeze_until)
        # Update param groups & LR
        self._reset_opt(start_lr)
        gamma = (end_lr / start_lr) ** (1 / (num_it - 1))
        scheduler = MultiplicativeLR(self.optimizer, lambda step: gamma)

        self.lr_recorder = [start_lr * gamma ** idx for idx in range(num_it)]
        self.loss_recorder = []

        for batch_idx, (x, target) in enumerate(self.train_loader):
            x, target = self.to_cuda(x, target)

            # Forward
            batch_loss = self._get_loss(x, target)
            self._backprop_step(batch_loss)
            # Update LR
            scheduler.step()

            # Record
            self.loss_recorder.append(batch_loss.item())
            # Stop after the number of iterations
            if batch_idx + 1 == num_it:
                break

    def plot_recorder(self, beta: float = 0.95, block: bool = True) -> None:
        """Display the results of the LR grid search

        Args:
            beta (float, optional): smoothing factor
            block (bool, optional): whether the plot should block execution
        """

        if len(self.lr_recorder) != len(self.loss_recorder) or len(self.lr_recorder) == 0:
            raise AssertionError("Please run the `lr_find` method first")

        # Exp moving average of loss
        smoothed_losses = []
        avg_loss = 0.
        for idx, loss in enumerate(self.loss_recorder):
            avg_loss = beta * avg_loss + (1 - beta) * loss
            smoothed_losses.append(avg_loss / (1 - beta ** (idx + 1)))

        plt.plot(self.lr_recorder[10:-5], smoothed_losses[10:-5])
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Training loss')
        plt.grid(True, linestyle='--', axis='x')
        plt.show(block=block)

    def check_setup(self, freeze_until: Optional[str] = None, lr: float = 3e-4, num_it: int = 100) -> bool:
        """Check whether you can overfit one batch

        Args:
            freeze_until (str, optional): last layer to freeze
            lr (float, optional): learning rate to be used for training
            num_it (int, optional): number of iterations to perform
        """

        self.model = freeze_model(self.model.train(), freeze_until)
        # Update param groups & LR
        self._reset_opt(lr)

        prev_loss = math.inf

        x, target = next(iter(self.train_loader))
        x, target = self.to_cuda(x, target)

        for _ in range(num_it):
            # Forward
            batch_loss = self._get_loss(x, target)
            # Backprop
            self._backprop_step(batch_loss)

            # Check that loss decreases
            if batch_loss.item() > prev_loss:
                return False
            prev_loss = batch_loss.item()

        return True


class ClassificationTrainer(Trainer):
    """Image classification trainer class

    Args:
        model (torch.nn.Module): model to train
        train_loader (torch.utils.data.DataLoader): training loader
        val_loader (torch.utils.data.DataLoader): validation loader
        criterion (torch.nn.Module): loss criterion
        optimizer (torch.optim.Optimizer): parameter optimizer
        gpu (int, optional): index of the GPU to use
        output_file (str, optional): path where checkpoints will be saved
    """

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the validation set

        Returns:
            dict: evaluation metrics
        """

        self.model.eval()

        val_loss, top1, top5, num_samples = 0., 0, 0, 0
        for x, target in self.val_loader:
            x, target = self.to_cuda(x, target)

            # Forward
            out = self.model(x)
            # Loss computation
            val_loss += self.criterion(out, target).item()

            pred = out.topk(5, dim=1)[1]
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            top1 += correct[:, 0].sum().item()
            top5 += correct.any(dim=1).sum().item()
            num_samples += x.shape[0]

        val_loss /= len(self.val_loader)

        return dict(val_loss=val_loss, acc1=top1 / num_samples, acc5=top5 / num_samples)

    @staticmethod
    def _eval_metrics_str(eval_metrics: Dict[str, float]) -> str:
        return (f"Validation loss: {eval_metrics['val_loss']:.4} "
                f"(Acc@1: {eval_metrics['acc1']:.2%}, Acc@5: {eval_metrics['acc5']:.2%})")


class SegmentationTrainer(Trainer):
    """Semantic segmentation trainer class

    Args:
        model (torch.nn.Module): model to train
        train_loader (torch.utils.data.DataLoader): training loader
        val_loader (torch.utils.data.DataLoader): validation loader
        criterion (torch.nn.Module): loss criterion
        optimizer (torch.optim.Optimizer): parameter optimizer
        gpu (int, optional): index of the GPU to use
        output_file (str, optional): path where checkpoints will be saved
    """

    @torch.no_grad()
    def evaluate(self, ignore_index: int = 255) -> Dict[str, float]:
        """Evaluate the model on the validation set

        Args:
            ignore_index (int, optional): index of the class to ignore in evaluation

        Returns:
            dict: evaluation metrics
        """

        self.model.eval()

        val_loss, mean_iou = 0., 0.
        for x, target in self.val_loader:
            x, target = self.to_cuda(x, target)

            # Forward
            out = self.model(x)
            # Loss computation
            val_loss += self.criterion(out, target).item()

            pred = out.argmax(dim=1)
            tmp_iou, num_seg = 0, 0
            for class_idx in torch.unique(target):
                if class_idx != ignore_index:
                    inter = (pred[target == class_idx] == class_idx).sum().item()
                    tmp_iou += inter / ((pred == class_idx) | (target == class_idx)).sum().item()
                    num_seg += 1
            mean_iou += tmp_iou / num_seg

        val_loss /= len(self.val_loader)
        mean_iou /= len(self.val_loader)

        return dict(val_loss=val_loss, mean_iou=mean_iou)

    @staticmethod
    def _eval_metrics_str(eval_metrics: Dict[str, float]) -> str:
        return f"Validation loss: {eval_metrics['val_loss']:.4} (Mean IoU: {eval_metrics['mean_iou']:.2%})"


def assign_iou(gt_boxes: Tensor, pred_boxes: Tensor, iou_threshold: float = 0.5) -> Tuple[List[int], List[int]]:
    """Assigns boxes by IoU"""
    iou = box_iou(gt_boxes, pred_boxes)
    iou = iou.max(dim=1)
    gt_kept = iou.values >= iou_threshold
    assign_unique = torch.unique(iou.indices[gt_kept])
    # Filter
    if iou.indices[gt_kept].shape[0] == assign_unique.shape[0]:
        return torch.arange(gt_boxes.shape[0])[gt_kept], iou.indices[gt_kept]  # type: ignore[return-value]
    else:
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
    """

    @staticmethod
    def _to_cuda(  # type: ignore[override]
        x: List[Tensor],
        target: List[Dict[str, Tensor]]
    ) -> Tuple[List[Tensor], List[Dict[str, Tensor]]]:
        """Move input and target to GPU"""
        x = [_x.cuda(non_blocking=True) for _x in x]
        target = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in target]
        return x, target

    def _backprop_step(self, loss: Tensor, grad_clip: float = .1) -> None:
        # Clean gradients
        self.optimizer.zero_grad()
        # Backpropate the loss
        loss.backward()
        # Safeguard for Gradient explosion
        if isinstance(grad_clip, float):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        # Update the params
        self.optimizer.step()

    def _get_loss(self, x: List[Tensor], target: List[Dict[str, Tensor]]) -> Tensor:  # type: ignore[override]
        # Forward & loss computation
        loss_dict = self.model(x, target)
        return sum(loss_dict.values())  # type: ignore[return-value]

    @staticmethod
    def _eval_metrics_str(eval_metrics: Dict[str, float]) -> str:
        return (f"Loc error: {eval_metrics['loc_err']:.2%} | Clf error: {eval_metrics['clf_err']:.2%} | "
                f"Det error: {eval_metrics['det_err']:.2%}")

    @torch.no_grad()
    def evaluate(self, iou_threshold: float = 0.5) -> Dict[str, float]:
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
            detections = self.model(x)

            for dets, t in zip(detections, target):
                if t['boxes'].shape[0] > 0 and dets['boxes'].shape[0] > 0:
                    gt_indices, pred_indices = assign_iou(t['boxes'], dets['boxes'], iou_threshold)
                    loc_assigns += len(gt_indices)
                    _correct = (t['labels'][gt_indices] == dets['labels'][pred_indices]).sum().item()
                else:
                    gt_indices, pred_indices = [], []
                    _correct = 0
                correct += _correct
                clf_error += len(gt_indices) - _correct
                loc_fn += t['boxes'].shape[0] - len(gt_indices)
                loc_fp += dets['boxes'].shape[0] - len(pred_indices)
            num_samples += sum(t['boxes'].shape[0] for t in target)

        nb_preds = num_samples - loc_fn + loc_fp
        # Localization
        loc_err = 1 - 2 * loc_assigns / (nb_preds + num_samples) if nb_preds + num_samples > 0 else 1.
        # Classification
        clf_err = 1 - correct / loc_assigns if loc_assigns > 0 else 1.
        # End-to-end
        det_err = 1 - 2 * correct / (nb_preds + num_samples) if nb_preds + num_samples > 0 else 1.
        return dict(loc_err=loc_err, clf_err=clf_err, det_err=det_err, val_loss=loc_err)
