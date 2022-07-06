import warnings
import torch
import math

from typing import Union


class LinearLR(torch.optim.lr_scheduler._LRScheduler):
    epsilon = 1e-12

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            lr_start: float = 0.0,
            lr_stop: float = 0.0,
            lr_steps: int = 1,
            last_epoch: int = -1,
            warmup: Union[int, float] = 0.0
    ):
        self.optimizer: torch.optim.Optimizer = optimizer

        self.lr_start: float = lr_start
        self.lr_stop: float = lr_stop if lr_stop > 0 else self.epsilon
        self.lr_steps: int = lr_steps
        self.current_lr_idx: int = 0
        self.warmup: Union[int, float] = warmup

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
            beta_min: float = 0.0,
            beta_max: float = 1.0,
            cycles: int = 1,
            warmup: Union[int, float] = 0.5
    ):
        assert not isinstance(warmup, float) or 0. < warmup <= 1.

        self.beta_min: float = beta_min
        self.beta_max: float = beta_max
        self.steps: int = steps
        self.cycles: int = cycles
        self.cycle_len: int = math.ceil(self.steps / self.cycles)
        self.warmup: int = warmup if warmup > 1 else math.ceil(warmup * self.cycle_len)

        self.m: float = (self.beta_max - self.beta_min) / self.warmup

        self.current_beta_idx: int = 0

    def get_beta(self) -> float:
        current_beta_idx = self.current_beta_idx
        self.current_beta_idx += 1

        if current_beta_idx >= self.steps:
            raise IndexError()

        return min(self.beta_min + ((current_beta_idx % self.cycle_len) * self.m), self.beta_max)
