'''Base Model for distance or bilinear based models

[h, r, t] -(get_embedding, optional to use a encoder like gcn)-> encoded_embeddings -(score_function)-> score

'''

from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray
import torch
import torch.nn as nn
from torch import Tensor

class KGModel(nn.Module, ABC):

    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def get_embeddings():
        pass
    
    @abstractmethod
    def score_func():
        pass

    @abstractmethod
    def get_reg(self, triples):
        pass

    def forward(self, triples, eval_model=False):

        # triples -> embedding (may contains an extra encoder like gnn)
        embeddings = self.get_embeddings(triples)
        # embedding -> score
        scores = self.score_func(embeddings)
        
        # get regularization terms
        reg = self.get_reg(triples)

        return scores, reg
        



        


