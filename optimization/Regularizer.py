'''Regularization temrs'''

from typing import Tuple
from abc import abstractmethod, ABC
import torch
import torch.nn as nn
from torch import Tensor
from utils.calculation import givens_rotation, quaternion_rotation

ALL_REGULARIZER = ["F2", "DURA_RESCAL", "DURA_W", "DURA_UniBi_2", "DURA_QuatE"] # log for all regs

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
        
        for factor in factors:
            norm += torch.pow(factor, 2).sum()

        # TODO try this one instead 
        # norm = torch.pow(factors, 2).sum()
        
        return  self.weight * norm / factors[0].shape[0]

class DURA_RESCAL(Regularizer):
    '''dura for rescal'''

    def __init__(self, weight: float) -> None:
        super().__init__(weight)
    
    def forward(self, factors: Tuple[Tensor]):

        norm = 0
        h, r, t = factors
        norm += torch.sum(t**2 + h**2)
        norm += torch.sum(torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2 + torch.bmm(r, t.unsqueeze(-1)) ** 2)

        return self.weight * norm / h.shape[0] 

class DURA_W(Regularizer):
    '''dura for complex'''
    def __init__(self, weight: float):
        super().__init__(weight)
    
    def forward(self, factors: Tuple[torch.Tensor]):
        norm = 0
        h, r, t = factors
        norm += 0.5 * torch.sum(t**2 + h**2)
        norm += 1.5 * torch.sum(h**2 * r**2 +  t**2 *  r**2)
        return self.weight * norm / h.shape[0]

class DURA_UniBi_2(Regularizer):
    def __init__(self, weight: float):
        super().__init__(weight)
        self.weight = weight
    
    def forward(self, factors):
        norm = 0
        h, Rot_u, Rel_s, Rot_v, t = factors # they have been processed
        uh = givens_rotation(Rot_u, h)
        suh = Rel_s * uh

        vt = givens_rotation(Rot_v, t, transpose=True)
        svt = Rel_s * vt

        norm += torch.sum(
            suh ** 2 + svt ** 2 + h ** 2 + t ** 2
        )

        return self.weight * norm / h.shape[0]


class DURA_QuatE(Regularizer):
    def __init__(self, weight: float):
        super().__init__(weight)
        self.weight = weight
    
    def forward(self, factors):
        norm = 0
        h, r, t = factors
        hr = quaternion_rotation(r, h, right=True)
        rt = quaternion_rotation(r, t, right=True, transpose=True)
        norm += torch.sum(hr ** 2 + rt ** 2 + h ** 2 + t ** 2)
        return self.weight * norm / h.shape[0]