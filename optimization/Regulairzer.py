'''Regularization temrs'''

from typing import Tuple
from abc import abstractmethod, ABC
import torch
import torch.nn as nn
from torch import Tensor

class Regularizer(nn.Module, ABC):
    '''Base class of all regularizer'''

    def __init__(self, weight:float) -> None:
        super().__init__()
        self.weight = weight

    @abstractmethod
    def forward(self, factors: Tuple[Tensor]):
        pass

class F2(Regularizer):
    '''Frobinues norm, default setting'''

    def __init__(self, weight: float) -> None:
        super().__init__(weight)
    
    def forward(self, factors: Tuple[Tensor]):

        norm = 0 
        h, r, t = factors

        norm = self.weight * torch.pow(r, 2).sum()
        return norm / factors[0].shape[0]