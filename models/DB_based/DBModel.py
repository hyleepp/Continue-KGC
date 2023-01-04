'''Base Model for distance or bilinear based models
'''

from numpy import ndarray
import torch 
import torch.nn as nn
from torch import Tensor 

from ..KGModel import KGModel

class DBModel(KGModel):

    def __init__(self, n_ent, n_rel, hidden_size) -> None:

        super().__init__(n_ent, n_rel, hidden_size)

        # naive version of embeddings
        self.emb_ent = nn.Embedding(n_ent, hidden_size)
        self.emb_rel = nn.Embedding(n_rel, hidden_size)

        return
    
    
    def get_embeddings(self, triples: ndarray) -> Tensor:

        emb_h = self.emb_ent[triples[0]]
        emb_r = self.emb_rel[triples[1]]
        emb_t = self.emb_ent[triples[2]]

        return emb_h, emb_r, emb_t

    # both score_func and get_reg are need to be implemented in child classes

    
    
        
    
    

        

    
    

        
    
