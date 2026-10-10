# Copyright (C) 2019-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, MultiplicativeLR, OneCycleLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .utils import freeze_bn, freeze_model, resolve_device, split_normalization_params

ParamSeq = Sequence[torch.nn.Parameter]

__all__ = ["Trainer"]


class Trainer:
    """Baseline trainer class.

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
        device: CPU, CUDA, MPS, or auto; cannot be combined with gpu
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        gpu: int | None = None,
        output_file: str = "./checkpoint.pth",
        amp: bool = False,
        skip_nan_loss: bool = False,
        nan_tolerance: int = 5,
        gradient_acc: int = 1,
        gradient_clip: float | None = None,
        on_epoch_end: Callable[[dict[str, float]], Any] | None = None,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.amp = amp
        self.scaler: GradScaler
        self.on_epoch_end = on_epoch_end
        self.skip_nan_loss = skip_nan_loss
        self.nan_tolerance = nan_tolerance
        self.gradient_acc = gradient_acc
        self.grad_clip = gradient_clip

        # Output file
        self.output_file = output_file

        # Initialize
        self.step = 0
        self.start_epoch = 0
        self.epoch = 0
        self._grad_count = 0
        self.min_loss = math.inf
        self._params: tuple[ParamSeq, ParamSeq] = ([], [])
        self.lr_recorder: list[float] = []
        self.loss_recorder: list[float] = []
        self.set_device(gpu, device=device)
        self._reset_opt(self.optimizer.defaults["lr"])

    def set_device(self, gpu: int | None = None, *, device: str | torch.device | None = None) -> None:
        """Move the model and criterion to CPU, CUDA, or MPS.

        Args:
            gpu: index of the target GPU device
            device: explicit device or auto; legacy gpu=None selects CPU

        Raises:
            ValueError: if both gpu and device are specified
        """
        if gpu is not None and device is not None:
            raise ValueError("Specify either gpu or device, not both")
        self.device = resolve_device(gpu if device is None else device)
        self.gpu = self.device.index if self.device.type == "cuda" else None
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.model = self.model.to(self.device)
        if isinstance(self.criterion, nn.Module):
            self.criterion = self.criterion.to(self.device)

    def save(self, output_file: str) -> None:
        """Save a trainer checkpoint

        Args:
            output_file: destination file path
        """
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": self.epoch,
                "step": self.step,
                "min_loss": self.min_loss,
                "model": self.model.state_dict(),
            },
            output_file,
            _use_new_zipfile_serialization=False,
        )

    def load(self, state: dict[str, Any]) -> None:
        """Resume from a trainer state

        Args:
            state: checkpoint dictionary
        """
        self.start_epoch = state["epoch"]
        self.epoch = self.start_epoch
        self.step = state["step"]
        self.min_loss = state["min_loss"]
        self.model.load_state_dict(state["model"])

    def _fit_epoch(self) -> None:
        """Fit a single epoch

        Raises:
            ValueError: if the loss value is NaN or inf
        """
        freeze_bn(self.model.train())

        nan_cnt = 0

        pb = tqdm(self.train_loader, desc="Training", leave=False)
        for x, target in pb:
            x, target = self.to_device(x, target)

            # Forward
            batch_loss: Tensor = self._get_loss(x, target)  # type: ignore[assignment]

            # Backprop
            if not self.skip_nan_loss or torch.isfinite(batch_loss):
                nan_cnt = 0
                if self._backprop_step(batch_loss):
                    self.scheduler.step()
            else:
                nan_cnt += 1
                if nan_cnt > self.nan_tolerance:
                    raise ValueError(f"loss value has been NaN or inf for more than {self.nan_tolerance} steps.")
            pb.set_postfix_str(f"Training loss: {batch_loss.item():.4}", refresh=False)

            self.step += 1
        if self._optimizer_step():
            self.scheduler.step()
        self.epoch += 1

    def to_cuda(
        self, x: Tensor, target: Tensor | list[dict[str, Tensor]]
    ) -> tuple[Tensor, Tensor | list[dict[str, Tensor]]]:
        """Compatibility alias for moving input and target to the selected device.

        Args:
            x: input tensor
            target: target tensor or list of target dictionaries

        Returns:
            tuple of input and target tensors

        """
        return self.to_device(x, target)

    def to_device(
        self, x: Tensor, target: Tensor | list[dict[str, Tensor]]
    ) -> tuple[Tensor, Tensor | list[dict[str, Tensor]]]:
        """Move input and target tensors to the selected device.

        Args:
            x: input tensor
            target: class, pixel, or detection targets

        Returns:
            Inputs and targets on the trainer device.
        """
        x = x.to(self.device, non_blocking=self.device.type == "cuda")
        target = target.to(self.device, non_blocking=self.device.type == "cuda")  # type: ignore[union-attr]
        return x, target

    def _backprop_step(self, loss: Tensor, force: bool = False) -> bool:
        self._grad_count += 1
        if self.amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        if self._grad_count < self.gradient_acc and not force:
            return False
        return self._optimizer_step()

    def _optimizer_step(self) -> bool:
        if self._grad_count == 0:
            return False

        if self.amp:
            self.scaler.unscale_(self.optimizer)
        if self._grad_count > 1:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.div_(self._grad_count)
        if isinstance(self.grad_clip, float):
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        if self.amp:
            scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            stepped = self.scaler.get_scale() >= scale
        else:
            self.optimizer.step()
            stepped = True
        self.optimizer.zero_grad(set_to_none=True)
        self._grad_count = 0
        return stepped

    def _get_loss(self, x: Tensor, target: Tensor, return_logits: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        # AMP
        if self.amp:
            with torch.amp.autocast(self.device.type):
                # Forward
                out = self.model(x)
                # Loss computation
                loss = cast(Tensor, self.criterion(out, target))
                if return_logits:
                    return loss, out
                return loss

        # Forward
        out = self.model(x)
        loss = cast(Tensor, self.criterion(out, target))
        if return_logits:
            return loss, out
        return loss

    def _set_params(self, norm_weight_decay: float | None = None) -> None:
        if not any(p.requires_grad for p in self.model.parameters()):
            raise AssertionError("All parameters are frozen")

        if norm_weight_decay is None:
            self._params = [p for p in self.model.parameters() if p.requires_grad], []
        else:
            self._params = split_normalization_params(self.model)

    def _reset_opt(self, lr: float, norm_weight_decay: float | None = None) -> None:
        """Reset the target params of the optimizer"""
        self.optimizer.defaults["lr"] = lr
        self.optimizer.state = defaultdict(dict)
        self.optimizer.param_groups = []
        self._set_params(norm_weight_decay)
        # Split it if norm layers needs custom WD
        if norm_weight_decay is None:
            self.optimizer.add_param_group({"params": self._params[0]})
        else:
            wd_groups = [norm_weight_decay, self.optimizer.defaults.get("weight_decay", 0)]
            for params, wd in zip(self._params, wd_groups, strict=True):
                if len(params) > 0:
                    self.optimizer.add_param_group({"params": params, "weight_decay": wd})
        self.optimizer.zero_grad(set_to_none=True)
        self._grad_count = 0

    @torch.inference_mode()
    def evaluate(self):  # type: ignore[no-untyped-def]  # noqa: D102, ANN201
        raise NotImplementedError

    @staticmethod
    def _eval_metrics_str(eval_metrics) -> str:  # type: ignore[no-untyped-def]  # noqa: ANN001
        raise NotImplementedError

    def _reset_scheduler(self, lr: float, num_epochs: int, sched_type: str = "onecycle", **kwargs: Any) -> None:
        self.scheduler: LRScheduler
        num_steps = num_epochs * math.ceil(len(self.train_loader) / self.gradient_acc)
        if sched_type == "onecycle":
            self.scheduler = OneCycleLR(self.optimizer, lr, num_steps, **kwargs)
        elif sched_type == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, num_steps, **kwargs)
        else:
            raise ValueError(f"The following scheduler type is not supported: {sched_type}")

    def fit_n_epochs(
        self,
        num_epochs: int,
        lr: float,
        freeze_until: str | None = None,
        sched_type: str = "onecycle",
        norm_weight_decay: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Train the model for a given number of epochs.

        Args:
            num_epochs: number of epochs to train
            lr: learning rate to be used by the scheduler
            freeze_until: last layer to freeze
            sched_type: type of scheduler to use
            norm_weight_decay: weight decay to apply to normalization parameters
            **kwargs: keyword args passed to the [`LRScheduler`][torch.optim.lr_scheduler.LRScheduler]
        """
        freeze_model(self.model.train(), freeze_until)
        # Update param groups & LR
        self._reset_opt(lr, norm_weight_decay)
        # Scheduler
        self._reset_scheduler(lr, num_epochs, sched_type, **kwargs)

        if self.amp:
            self.scaler = GradScaler(self.device.type, enabled=self.device.type != "cpu")

        for _ in tqdm(range(num_epochs), desc="Epochs"):
            self._fit_epoch()
            eval_metrics = self.evaluate()

            tqdm.write(f"Epoch {self.epoch}/{self.start_epoch + num_epochs} - {self._eval_metrics_str(eval_metrics)}")

            if eval_metrics["val_loss"] < self.min_loss:
                print(  # noqa: T201
                    f"Validation loss decreased {self.min_loss:.4} --> {eval_metrics['val_loss']:.4}: saving state..."
                )
                self.min_loss = eval_metrics["val_loss"]
                self.save(self.output_file)

            if self.on_epoch_end is not None:
                self.on_epoch_end(eval_metrics)

    def find_lr(
        self,
        freeze_until: str | None = None,
        start_lr: float = 1e-7,
        end_lr: float = 1,
        norm_weight_decay: float | None = None,
        num_it: int = 100,
    ) -> None:
        """Gridsearch the optimal learning rate for the training as described in
        ["Cyclical Learning Rates for Training Neural Networks"](https://arxiv.org/pdf/1506.01186.pdf).

        Args:
           freeze_until: last layer to freeze
           start_lr: initial learning rate
           end_lr: final learning rate
           norm_weight_decay: weight decay to apply to normalization parameters
           num_it: number of microbatches represented by the sweep; AMP overflow retries may consume more

        Raises:
            ValueError: if the number of iterations is greater than the number of available batches
            RuntimeError: if AMP overflows exhaust the available batches before the sweep completes
        """
        if num_it > len(self.train_loader):
            raise ValueError("the value of `num_it` needs to be lower than the number of available batches")

        freeze_model(self.model.train(), freeze_until)
        # Update param groups & LR
        self._reset_opt(start_lr, norm_weight_decay)
        num_steps = math.ceil(num_it / self.gradient_acc)
        gamma = (end_lr / start_lr) ** (1 / (num_steps - 1)) if num_steps > 1 else 1
        scheduler = MultiplicativeLR(self.optimizer, lambda step: gamma)

        self.lr_recorder = []
        self.loss_recorder = []

        if self.amp:
            self.scaler = GradScaler(self.device.type, enabled=self.device.type != "cpu")

        batch_iter = iter(self.train_loader)
        final_step_batches = num_it - self.gradient_acc * (num_steps - 1)
        for step_idx in tqdm(range(num_steps), total=num_steps, desc="Learning rates"):
            step_batches = final_step_batches if step_idx == num_steps - 1 else self.gradient_acc
            stepped = False
            while not stepped:
                accumulated_loss = 0.0
                for batch_idx in range(step_batches):
                    try:
                        x, target = next(batch_iter)
                    except StopIteration as exc:
                        raise RuntimeError("LR finder ran out of batches while recovering from AMP overflow") from exc
                    x, target = self.to_device(x, target)

                    # Forward
                    batch_loss: Tensor = self._get_loss(x, target)  # type: ignore[assignment]
                    if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                        if len(self.loss_recorder) == 0:
                            raise ValueError("loss value is NaN or inf.")
                        return

                    accumulated_loss += batch_loss.item()
                    stepped = self._backprop_step(batch_loss, force=batch_idx == step_batches - 1)

                if stepped:
                    self.lr_recorder.append(float(self.optimizer.param_groups[0]["lr"]))
                    self.loss_recorder.append(accumulated_loss / step_batches)
                    scheduler.step()

    def plot_recorder(self, beta: float = 0.95, **kwargs: Any) -> None:
        """Display the results of the LR grid search

        Args:
            beta: smoothing factor
            **kwargs: keyword args of [`matplotlib.pyplot.show`][matplotlib.pyplot.show]

        Raises:
            AssertionError: if the number of learning rate recorder and loss recorder are not the same or if the number of learning rate recorder is 0
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        if len(self.lr_recorder) != len(self.loss_recorder) or len(self.lr_recorder) == 0:
            raise AssertionError("Please run the `lr_find` method first")

        # Exp moving average of loss
        smoothed_losses = []
        avg_loss = 0.0
        for idx, loss in enumerate(self.loss_recorder):
            avg_loss = beta * avg_loss + (1 - beta) * loss
            smoothed_losses.append(avg_loss / (1 - beta ** (idx + 1)))

        plt.plot(self.lr_recorder, smoothed_losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate")
        plt.ylabel("Training loss")
        plt.grid(True, linestyle="--", axis="x")
        plt.show(**kwargs)

    def check_setup(
        self,
        freeze_until: str | None = None,
        lr: float = 3e-4,
        norm_weight_decay: float | None = None,
        num_it: int = 100,
        **kwargs: Any,
    ) -> None:
        """Check whether you can overfit one batch

        Args:
            freeze_until: last layer to freeze
            lr: learning rate to be used for training
            norm_weight_decay: weight decay to apply to normalization parameters
            num_it: number of iterations to perform
            **kwargs: keyword args of [`matplotlib.pyplot.show`][matplotlib.pyplot.show]

        Raises:
            ValueError: if the loss value is NaN or inf
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        freeze_model(self.model.train(), freeze_until)
        # Update param groups & LR
        self._reset_opt(lr, norm_weight_decay)

        x, target = next(iter(self.train_loader))
        x, target = self.to_device(x, target)

        losses = []

        if self.amp:
            self.scaler = GradScaler(self.device.type, enabled=self.device.type != "cpu")

        for idx in range(num_it):
            # Forward
            batch_loss: Tensor = self._get_loss(x, target)  # type: ignore[assignment]
            # Backprop
            self._backprop_step(batch_loss, force=idx + 1 == num_it)

            if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                raise ValueError("loss value is NaN or inf.")

            losses.append(batch_loss.item())

        plt.plot(range(len(losses)), losses)
        plt.xlabel("Optimization steps")
        plt.ylabel("Training loss")
        plt.grid(True, linestyle="--", axis="x")
        plt.show(**kwargs)
