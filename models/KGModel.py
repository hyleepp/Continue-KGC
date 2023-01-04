'''The base class for all specific models

[h, r, t] -(get_embedding, optional to use a encoder like gcn)-> encoded_embeddings -(score_function)-> score
'''

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from numpy import ndarray
import torch
import torch.nn as nn
from torch import Tensor

class KGModel(nn.Module, ABC):

    def __init__(self, n_ent, n_rel, hidden_size) -> None:
        super().__init__()
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.hidden_size = hidden_size

        return 
   
    @abstractmethod
    def get_embeddings(self, triples: ndarray) -> Tuple[Tensor, Tensor, Tensor]:
        """triples, one-hot vectors (h, r, t) -> embeddings, dense vectors

        Args:
            triples (np.ndarray): one hot vectors
        
        Returns:
            Tensor: the embeddings of all entities and relations in triples
            
        """
        pass
    
    @abstractmethod
    def score_func(self, emb_h: Tensor, emb_r: Tensor, emb_t: Tensor) -> Tensor:
        """calculate the score based on embeddings 

        Args:
            emb_h (Tensor): the embeddings of head entities 
            emb_r (Tensor): the embeddings of relations
            emb_t (Tensor): the embeddings of tail entities

        Returns:
            Tensor: the score of each triple
        """
        pass

    @abstractmethod
    def get_reg(self, triples: Tensor) -> Tensor:
        """calculate the regularization terms

        Args:
            triples (Tensor): input tirples.

        Returns:
            Tensor: the reg terms
        """
        pass


    def forward(self, triples: ndarray, eval_model=False) -> Tuple[Tensor, Tensor]:
        """forward function

        Args:
            triples (ndarray): input triples
            eval_model (bool, optional): whether or not use eval mode. Defaults to False.

        Returns:
            scores: the forward scores
            reg: reg terms
        """

        # triples -> embedding (may contains an extra encoder like gnn)
        emb_h, emb_r, emb_t = self.get_embeddings(triples)
        # embedding -> score
        scores = self.score_func(emb_h, emb_r, emb_t)
        
        # get regularization terms
        reg = self.get_reg(triples)

        return scores, reg
        



        