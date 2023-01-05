'''Implementations of basic distance based models'''

from DBModel import DBModel

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from ..utils.calculation import euc_distance


class TransE(DBModel):
    
    def __init__(self, n_ent, n_rel, hidden_size) -> None:
        super().__init__(n_ent, n_rel, hidden_size)
        
    def score_func(self, emb_h: Tensor, emb_r: Tensor, emb_t: Tensor, eval_mode=False) -> Tensor:
        emb_lhs = emb_h + emb_r
        return - euc_distance(emb_lhs, emb_t, eval_mode)




