import math
import warnings
from torch.optim.lr_scheduler import _LRScheduler


class MyLR(_LRScheduler):
    def __init__(self, optimizer, T_max, phase1_epoch, eta_min, last_epoch=-1, verbose=False, decay = 1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.phase1_epoch = phase1_epoch
        self.decay = decay
        self.base_lrs = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        elif self.last_epoch < self.phase1_epoch:
            return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                  (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                  (group['lr'] - self.eta_min) + self.eta_min
                  for group in self.optimizer.param_groups]
        else:
            return self.eta_min

    def _get_closed_form_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]

