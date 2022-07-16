# Copyright (C) 2019-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import math
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import ConsoleMasterBar
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiplicativeLR, OneCycleLR  # type: ignore[attr-defined]
from torch.utils.data import DataLoader

from .utils import freeze_bn, freeze_model, split_normalization_params

ParamSeq = Sequence[torch.nn.Parameter]  # type: ignore[name-defined]

__all__ = ["Trainer"]


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        gpu: Optional[int] = None,
        output_file: str = "./checkpoint.pth",
        amp: bool = False,
        skip_nan_loss: bool = False,
        nan_tolerance: int = 5,
        on_epoch_end: Optional[Callable[[Dict[str, float]], Any]] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.amp = amp
        self.scaler: torch.cuda.amp.grad_scaler.GradScaler
        self.on_epoch_end = on_epoch_end
        self.skip_nan_loss = skip_nan_loss
        self.nan_tolerance = nan_tolerance

        # Output file
        self.output_file = output_file

        # Initialize
        self.step = 0
        self.start_epoch = 0
        self.epoch = 0
        self.min_loss = math.inf
        self.gpu = gpu
        self._params: Tuple[ParamSeq, ParamSeq] = ([], [])
        self.lr_recorder: List[float] = []
        self.loss_recorder: List[float] = []
        self.set_device(gpu)
        self._reset_opt(self.optimizer.defaults["lr"])

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
        torch.save(
            dict(
                epoch=self.epoch,
                step=self.step,
                min_loss=self.min_loss,
                optimizer=self.optimizer.state_dict(),
                model=self.model.state_dict(),
            ),
            output_file,
            _use_new_zipfile_serialization=False,
        )

    def load(self, state: Dict[str, Any]) -> None:
        """Resume from a trainer state

        Args:
            state (dict): checkpoint dictionary
        """
        self.start_epoch = state["epoch"]
        self.epoch = self.start_epoch
        self.step = state["step"]
        self.min_loss = state["min_loss"]
        self.optimizer.load_state_dict(state["optimizer"])
        self.model.load_state_dict(state["model"])

    def _fit_epoch(self, mb: ConsoleMasterBar) -> None:
        """Fit a single epoch

        Args:
            mb (fastprogress.master_bar): primary progress bar
        """
        self.model = freeze_bn(self.model.train())

        nan_cnt = 0

        pb = progress_bar(self.train_loader, parent=mb)
        for x, target in pb:
            x, target = self.to_cuda(x, target)

            # Forward
            batch_loss: Tensor = self._get_loss(x, target)  # type: ignore[assignment]

            # Backprop
            if not self.skip_nan_loss or torch.isfinite(batch_loss):
                nan_cnt = 0
                self._backprop_step(batch_loss)
            else:
                nan_cnt += 1
                if nan_cnt > self.nan_tolerance:
                    raise ValueError(f"loss value has been NaN or inf for more than {self.nan_tolerance} steps.")
            # Update LR
            self.scheduler.step()
            pb.comment = f"Training loss: {batch_loss.item():.4}"

            self.step += 1
        self.epoch += 1

    def to_cuda(
        self, x: Tensor, target: Union[Tensor, List[Dict[str, Tensor]]]
    ) -> Tuple[Tensor, Union[Tensor, List[Dict[str, Tensor]]]]:
        """Move input and target to GPU"""
        if isinstance(self.gpu, int):
            if self.gpu >= torch.cuda.device_count():
                raise ValueError("Invalid device index")
            return self._to_cuda(x, target)  # type: ignore[arg-type]
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
        if self.amp:
            self.scaler.scale(loss).backward()
            # Update the params
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            # Update the params
            self.optimizer.step()

    def _get_loss(self, x: Tensor, target: Tensor, return_logits: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # AMP
        if self.amp:
            with torch.cuda.amp.autocast():  # type: ignore[attr-defined]
                # Forward
                out = self.model(x)
                # Loss computation
                loss = self.criterion(out, target)
                if return_logits:
                    return loss, out
                return loss

        # Forward
        out = self.model(x)
        loss = self.criterion(out, target)
        if return_logits:
            return loss, out
        return loss

    def _set_params(self, norm_weight_decay: Optional[float] = None) -> None:
        if not any(p.requires_grad for p in self.model.parameters()):
            raise AssertionError("All parameters are frozen")

        if norm_weight_decay is None:
            self._params = [p for p in self.model.parameters() if p.requires_grad], []
        else:
            self._params = split_normalization_params(self.model)

    def _reset_opt(self, lr: float, norm_weight_decay: Optional[float] = None) -> None:
        """Reset the target params of the optimizer"""
        self.optimizer.defaults["lr"] = lr
        self.optimizer.state = defaultdict(dict)
        self.optimizer.param_groups = []
        self._set_params(norm_weight_decay)
        # Split it if norm layers needs custom WD
        if norm_weight_decay is None:
            self.optimizer.add_param_group(dict(params=self._params[0]))
        else:
            wd_groups = [norm_weight_decay, self.optimizer.defaults.get("weight_decay", 0)]
            for _params, _wd in zip(self._params, wd_groups):
                if len(_params) > 0:
                    self.optimizer.add_param_group(dict(params=_params, weight_decay=_wd))

    @torch.inference_mode()
    def evaluate(self):
        raise NotImplementedError

    @staticmethod
    def _eval_metrics_str(eval_metrics):
        raise NotImplementedError

    def _reset_scheduler(self, lr: float, num_epochs: int, sched_type: str = "onecycle") -> None:
        if sched_type == "onecycle":
            self.scheduler = OneCycleLR(self.optimizer, lr, num_epochs * len(self.train_loader))
        elif sched_type == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, num_epochs * len(self.train_loader), eta_min=lr / 25e4)
        else:
            raise ValueError(f"The following scheduler type is not supported: {sched_type}")

    def fit_n_epochs(
        self,
        num_epochs: int,
        lr: float,
        freeze_until: Optional[str] = None,
        sched_type: str = "onecycle",
        norm_weight_decay: Optional[float] = None,
    ) -> None:
        """Train the model for a given number of epochs

        Args:
            num_epochs (int): number of epochs to train
            lr (float): learning rate to be used by the scheduler
            freeze_until (str, optional): last layer to freeze
            sched_type (str, optional): type of scheduler to use
            norm_weight_decay (float, optional): weight decay to apply to normalization parameters
        """

        self.model = freeze_model(self.model.train(), freeze_until)
        # Update param groups & LR
        self._reset_opt(lr, norm_weight_decay)
        # Scheduler
        self._reset_scheduler(lr, num_epochs, sched_type)

        if self.amp:
            self.scaler = torch.cuda.amp.GradScaler()  # type: ignore[attr-defined]

        mb = master_bar(range(num_epochs))
        for _ in mb:

            self._fit_epoch(mb)
            eval_metrics = self.evaluate()

            # master bar
            mb.main_bar.comment = f"Epoch {self.epoch}/{self.start_epoch + num_epochs}"
            mb.write(f"Epoch {self.epoch}/{self.start_epoch + num_epochs} - " f"{self._eval_metrics_str(eval_metrics)}")

            if eval_metrics["val_loss"] < self.min_loss:
                print(
                    f"Validation loss decreased {self.min_loss:.4} --> "
                    f"{eval_metrics['val_loss']:.4}: saving state..."
                )
                self.min_loss = eval_metrics["val_loss"]
                self.save(self.output_file)

            if self.on_epoch_end is not None:
                self.on_epoch_end(eval_metrics)

    def find_lr(
        self,
        freeze_until: Optional[str] = None,
        start_lr: float = 1e-7,
        end_lr: float = 1,
        norm_weight_decay: Optional[float] = None,
        num_it: int = 100,
    ) -> None:
        """Gridsearch the optimal learning rate for the training

        Args:
           freeze_until (str, optional): last layer to freeze
           start_lr (float, optional): initial learning rate
           end_lr (float, optional): final learning rate
           norm_weight_decay (float, optional): weight decay to apply to normalization parameters
           num_it (int, optional): number of iterations to perform
        """

        if num_it > len(self.train_loader):
            raise ValueError("the value of `num_it` needs to be lower than the number of available batches")

        self.model = freeze_model(self.model.train(), freeze_until)
        # Update param groups & LR
        self._reset_opt(start_lr, norm_weight_decay)
        gamma = (end_lr / start_lr) ** (1 / (num_it - 1))
        scheduler = MultiplicativeLR(self.optimizer, lambda step: gamma)

        self.lr_recorder = [start_lr * gamma**idx for idx in range(num_it)]
        self.loss_recorder = []

        if self.amp:
            self.scaler = torch.cuda.amp.GradScaler()  # type: ignore[attr-defined]

        for batch_idx, (x, target) in enumerate(self.train_loader):
            x, target = self.to_cuda(x, target)

            # Forward
            batch_loss: Tensor = self._get_loss(x, target)  # type: ignore[assignment]
            self._backprop_step(batch_loss)
            # Update LR
            scheduler.step()

            # Record
            if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                if batch_idx == 0:
                    raise ValueError("loss value is NaN or inf.")
                else:
                    break
            self.loss_recorder.append(batch_loss.item())
            # Stop after the number of iterations
            if batch_idx + 1 == num_it:
                break

        self.lr_recorder = self.lr_recorder[: len(self.loss_recorder)]

    def plot_recorder(self, beta: float = 0.95, **kwargs: Any) -> None:
        """Display the results of the LR grid search

        Args:
            beta (float, optional): smoothing factor
        """

        if len(self.lr_recorder) != len(self.loss_recorder) or len(self.lr_recorder) == 0:
            raise AssertionError("Please run the `lr_find` method first")

        # Exp moving average of loss
        smoothed_losses = []
        avg_loss = 0.0
        for idx, loss in enumerate(self.loss_recorder):
            avg_loss = beta * avg_loss + (1 - beta) * loss
            smoothed_losses.append(avg_loss / (1 - beta ** (idx + 1)))

        # Properly rescale Y-axis
        data_slice = slice(
            min(len(self.loss_recorder) // 10, 10),
            -min(len(self.loss_recorder) // 20, 5) if len(self.loss_recorder) >= 20 else len(self.loss_recorder),
        )
        vals: np.ndarray = np.array(smoothed_losses[data_slice])
        min_idx = vals.argmin()
        max_val = vals.max() if min_idx is None else vals[: min_idx + 1].max()  # type: ignore[misc]
        delta = max_val - vals[min_idx]

        plt.plot(self.lr_recorder[data_slice], smoothed_losses[data_slice])
        plt.xscale("log")
        plt.xlabel("Learning Rate")
        plt.ylabel("Training loss")
        plt.ylim(vals[min_idx] - 0.1 * delta, max_val + 0.2 * delta)
        plt.grid(True, linestyle="--", axis="x")
        plt.show(**kwargs)

    def check_setup(
        self,
        freeze_until: Optional[str] = None,
        lr: float = 3e-4,
        norm_weight_decay: Optional[float] = None,
        num_it: int = 100,
    ) -> bool:
        """Check whether you can overfit one batch

        Args:
            freeze_until (str, optional): last layer to freeze
            lr (float, optional): learning rate to be used for training
            norm_weight_decay (float, optional): weight decay to apply to normalization parameters
            num_it (int, optional): number of iterations to perform
        """

        self.model = freeze_model(self.model.train(), freeze_until)
        # Update param groups & LR
        self._reset_opt(lr, norm_weight_decay)

        x, target = next(iter(self.train_loader))
        x, target = self.to_cuda(x, target)

        _losses = []

        if self.amp:
            self.scaler = torch.cuda.amp.GradScaler()  # type: ignore[attr-defined]

        for _ in range(num_it):
            # Forward
            batch_loss: Tensor = self._get_loss(x, target)  # type: ignore[assignment]
            # Backprop
            self._backprop_step(batch_loss)

            if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                raise ValueError("loss value is NaN or inf.")

            _losses.append(batch_loss.item())

        return _losses[-1] < _losses[0]
