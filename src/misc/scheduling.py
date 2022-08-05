import warnings
import torch
import math

from typing import Union, Optional


class LinearLR(torch.optim.lr_scheduler._LRScheduler):
    epsilon = 1e-12

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            lr: float,
            lr_start: Optional[float] = None,
            lr_stop: Optional[float] = None,
            lr_steps: int = 1,
            last_epoch: int = -1,
            warmup: Union[int, float] = 0
    ):
        assert isinstance(warmup, int) or 0. < warmup <= 1.

        self.optimizer: torch.optim.Optimizer = optimizer

        self.lr: float = lr
        self.lr_start: float = lr_start if lr_start is not None else lr
        self.lr_start = self.lr_start if self.lr_start > 0 else self.epsilon
        self.lr_stop: float = lr_stop if lr_stop is not None else lr
        self.lr_stop = self.lr_stop if self.lr_stop > 0 else self.epsilon
        self.lr_steps: int = lr_steps
        self.current_lr_idx: int = 0
        self.warmup: int = warmup if isinstance(warmup, int) else math.ceil(warmup * self.lr_steps)

        if self.warmup > 0:
            self.lr_values = torch.cat([
                torch.linspace(self.lr_start, self.lr, self.warmup),
                torch.linspace(self.lr, self.lr_stop, self.lr_steps - self.warmup)
            ])
        else:
            self.lr_values = torch.linspace(self.lr, self.lr_stop, self.lr_steps)

        super(LinearLR, self).__init__(optimizer, last_epoch=last_epoch)

    def state_dict(self):
        warnings.warn(torch.optim.lr_scheduler.SAVE_STATE_WARNING, UserWarning)
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer',)}

        return state_dict

    def load_state_dict(self, state_dict):
        warnings.warn(torch.optim.lr_scheduler.SAVE_STATE_WARNING, UserWarning)

        self.lr_start = state_dict.pop('lr_start')
        self.lr_stop = state_dict.pop('lr_stop')
        self.lr_steps = state_dict.pop('lr_steps')
        self.current_lr_idx = state_dict.pop('current_lr_idx')
        self.warmup = state_dict.pop('warmup')

        if 0.0 < self.warmup < 1.:
            self.lr_values = torch.cat([
                torch.linspace(self.epsilon, self.lr_start, int(self.lr_steps * self.warmup)),
                torch.linspace(self.lr_start, self.lr_stop, self.lr_steps - int(self.lr_steps * self.warmup))
            ])
        elif self.warmup >= 1:
            self.lr_values = torch.cat([
                torch.linspace(self.epsilon, self.lr_start, self.warmup),
                torch.linspace(self.lr_start, self.lr_stop, self.lr_steps - self.warmup)
            ])
        else:
            self.lr_values = torch.linspace(self.lr_start, self.lr_stop, self.lr_steps)

        self.__dict__.update(state_dict)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.")

        current_lr_idx = self.current_lr_idx
        self.current_lr_idx += 1

        try:
            lr = self.lr_values[current_lr_idx]
        except IndexError:
            lr = 0.0

        return [lr]


class BetaCyclicalAnnealer:
    def __init__(
            self,
            steps: int,
            beta: float = 1.0,
            beta_start: Optional[float] = None,
            beta_stop: Optional[float] = None,
            cycles: int = 0,
            warmup: Union[int, float] = 0.5
    ):
        assert isinstance(warmup, int) or 0. < warmup <= 1.

        self.beta: float = beta
        self.beta_start: float = beta_start if beta_start is not None else beta
        self.beta_stop: float = beta_stop if beta_stop is not None else beta
        self.steps: int = steps
        self.cycles: int = cycles
        if cycles > 0:  # If there are no cycles the schedule is constant
            self.cycle_len: int = math.ceil(self.steps / self.cycles)
            self.warmup: int = warmup if warmup > 1 else math.ceil(warmup * self.cycle_len)
            if warmup > 0:  # If there is no warmup or there are no cycles the schedule is constant
                self.m: float = (self.beta_stop - self.beta_start) / self.warmup
        self.current_beta_idx: int = 0

    def get_beta(self) -> float:
        self.current_beta_idx

        if self.current_beta_idx > self.steps:
            raise IndexError()

        try:
            beta = min(self.beta_start + ((self.current_beta_idx % self.cycle_len) * self.m), self.beta_stop)
        except AttributeError:
            beta = self.beta
        return beta

    def step(self):
        self.current_beta_idx += 1
