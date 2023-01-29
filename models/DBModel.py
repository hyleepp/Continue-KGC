'''Base Model for distance or bilinear based models
'''

from numpy import ndarray
import torch 
import torch.nn as nn
from torch import Tensor 

from .KGModel import KGModel
from .utils.calculation import euc_distance, givens_rotation

DISTANCE_MODELS = ['TransE', 'RotatE', 'RotE']
BILINEAR_MODELS = ['RESCAL']

class DBModel(KGModel):

    def __init__(self, args) -> None:

        super().__init__(args)

        self.similarity_method = 'dist' # or dot 

        return
    
    def encode(self, triples: Tensor) -> Tensor:

        # we do not need this part, it is used in GNN based models
        return self.emb_ent.weight, self.emb_rel.weight # here we need tensor, rather than Embedding

    def score(self, v_queries, v_candidates, eval_mode=False) -> Tensor:

        # TODO continue here
        if self.similarity_method == 'dist':
            score = - euc_distance(v_queries, v_candidates, eval_mode)
        elif self.similarity_method == 'dot':
            score = v_queries @ v_candidates.transpose(0, 1) if eval_mode \
                    else torch.sum(v_queries * v_candidates, dim=-1, keepdim=True)
        else:
            raise KeyError(f"The given method {self.similarity_method} is not implemented.")
        
        return score

    # both score_func and get_reg are need to be implemented in child classes

    
class TransE(DBModel):
    
    def __init__(self, args) -> None:
        super().__init__(args)

    def get_queries(self, triples, enc_e, enc_r) -> Tensor:

        h = enc_e[triples[:, 0]]
        r = enc_r[triples[:, 1]]

        return h + r
    
    
class RotatE(DBModel):

    def __init__(self, args) -> None:
        super().__init__(args)
    
    def get_queries(self, triples, enc_e, enc_r) -> Tensor:

        h = enc_e[triples[:, 0]]
        r = enc_r[triples[:, 1]]
        hr = givens_rotation(r, h)

        return hr
    
class RotE(DBModel):
    '''rotate + trans'''

    def __init__(self, args) -> None:

        super().__init__(args)
        self.emb_trans = nn.Embedding(self.n_rel * 2, self.hidden_size, device=args.device)
        if args.init_scale > 0:
            self.emb_trans.weight.data = args.init_scale * torch.randn((self.n_rel * 2, self.hidden_size))

        return
    
    def encode(self, triples: Tensor) -> Tensor:
        return self.emb_ent.weight, (self.emb_rel.weight, self.emb_trans.weight) # here we need tensor, rather than Embedding
    
    def get_queries(self, triples, enc_e, enc_r) -> Tensor:

        rot_r, trans_r = enc_r # enc_r contains two part, rotation and translation

        h = enc_e[triples[:, 0]]
        rot_r = rot_r[triples[:, 1]]
        trans_r = trans_r[triples[:, 1]]
        hr = givens_rotation(rot_r, h) + trans_r

        return hr
    
            


        