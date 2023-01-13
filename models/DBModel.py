'''Base Model for distance or bilinear based models
'''

from numpy import ndarray
import torch 
import torch.nn as nn
from torch import Tensor 

from .KGModel import KGModel
from .utils.calculation import euc_distance

DISTANCE_MODELS = ['TransE']
BILINEAR_MODELS = ['RESCAL']

class DBModel(KGModel):

    def __init__(self, args) -> None:

        super().__init__(args)

        # naive version of embeddings
        self.emb_ent = nn.Embedding(self.n_ent, self.hidden_size)
        self.emb_rel = nn.Embedding(self.n_rel * 2, self.hidden_size) # introduce reciprocal relations
        self.similarity_method = 'dist' # or dot 

        return
    
    def get_candidates(self, triples, emb_e) -> Tensor:

        t = triples[:, 2]
        candidates = emb_e[t]
        
        return candidates
    
    def encode(self, triples: Tensor) -> Tensor:

        # we do not need this part, it is used in GNN based models
        return self.emb_ent, self.emb_rel

    def score(self, emb_queries, emb_candidates, eval_mode=False) -> Tensor:

        # TODO continue here
        if self.similarity_method == 'dist':
            score = - euc_distance(emb_queries, emb_candidates, eval_mode)
        elif self.similarity_method == 'dot':
            score = emb_queries @ emb_candidates.transpose(0, 1) if eval_mode \
                    else torch.sum(emb_queries * emb_candidates, dim=-1, keepdim=True)

        else:
            raise KeyError(f"The given method {self.similarity_method} is not implemented.")

    # both score_func and get_reg are need to be implemented in child classes

    
class TransE(DBModel):
    
    def __init__(self, args) -> None:
        super().__init__(args)

    def get_queries(self, triples, emb_e, emb_r) -> Tensor:

        h = emb_e[triples[:, 0]]
        r = emb_r[triples[:, 1]]

        return h + r
    
    