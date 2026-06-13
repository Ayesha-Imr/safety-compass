from __future__ import annotations

from typing import Optional

import torch
from transformers import TrainerCallback

from safety_compass.logger import CompassCSVLogger


class SafetyCompassCallback(TrainerCallback):
    """Hugging Face Trainer callback for periodic Safety Compass measurements."""

    def __init__(
        self,
        monitor,
        measure_every_n_steps: int = 100,
        log_file: str = "safety_compass_log.csv",
        logger: Optional[CompassCSVLogger] = None,
        include_step_zero: bool = True,
        include_train_end: bool = True,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
    ):
        self.monitor = monitor
        self.measure_every_n_steps = int(measure_every_n_steps)
        self.logger = logger or CompassCSVLogger(
            log_file,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
        )
        self.include_step_zero = include_step_zero
        self.include_train_end = include_train_end
        self._logged_steps = set()

    def _measure_and_log(self, state, model=None):
        if model is not None:
            self.monitor.set_model(model)

        active_model = model or self.monitor.model
        was_training = bool(getattr(active_model, "training", False))
        active_model.eval()

        with torch.no_grad():
            if not self.monitor.is_setup:
                self.monitor.setup()
            row = self.monitor.measure(
                step=int(state.global_step),
                epoch=getattr(state, "epoch", None),
            )

        if was_training:
            active_model.train()

        self.logger.log(row)
        self._logged_steps.add(int(state.global_step))

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.include_step_zero and int(state.global_step) not in self._logged_steps:
            self._measure_and_log(state, model=model)
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step)
        if step <= 0:
            return control
        if step in self._logged_steps:
            return control
        if step % self.measure_every_n_steps == 0:
            self._measure_and_log(state, model=model)
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step)
        if self.include_train_end and step not in self._logged_steps:
            self._measure_and_log(state, model=model)
        self.logger.close()
        return control
