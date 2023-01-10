'''Base Model for distance or bilinear based models
'''

from numpy import ndarray
import torch 
import torch.nn as nn
from torch import Tensor 

from KGModel import KGModel
from utils.calculation import euc_distance

DISTANCE_MODELS = ['TransE']
BILINEAR_MODELS = ['RESCAL']

class DBModel(KGModel):

    def __init__(self, args) -> None:

        super().__init__(args)

        # naive version of embeddings
        self.emb_ent = nn.Embedding(self.n_ent, self.hidden_size)
        self.emb_rel = nn.Embedding(self.n_rel, self.hidden_size)

        return
    
    
    def get_embeddings(self, triples: ndarray) -> Tensor:

        emb_h = self.emb_ent[triples[0]]
        emb_r = self.emb_rel[triples[1]]
        emb_t = self.emb_ent[triples[2]]

        return emb_h, emb_r, emb_t

    # both score_func and get_reg are need to be implemented in child classes

    
class TransE(DBModel):
    
    def __init__(self, args) -> None:
        super().__init__(args)
        
    def score_func(self, emb_h: Tensor, emb_r: Tensor, emb_t: Tensor, eval_mode=False) -> Tensor:
        emb_lhs = emb_h + emb_r
        return - euc_distance(emb_lhs, emb_t, eval_mode)
    