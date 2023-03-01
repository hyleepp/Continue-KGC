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

    def score(self, v_queries, v_candidates, eval_mode=False, require_grad=True) -> Tensor:

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
    
class ComplEx(DBModel):

    def __init__(self, args) -> None:
        super().__init__(args)

    def get_queries(self, triples, enc_e, enc_r) -> Tensor:
        pass

class RESCAL(DBModel):

    def __init__(self, args) -> None:
        super().__init__(args)
        self.similarity_method = 'dot'
        self.emb_rel = nn.Embedding(self.n_rel * 2, self.hidden_size * self.hidden_size)
        if args.init_scale > 0:
            self.emb_rel.weight.data = args.init_scale * torch.randn((self.n_rel * 2, self.hidden_size * self.hidden_size))
        
    def get_queries(self, triples, enc_e, enc_r) -> Tensor:
        h = enc_e[triples[:, 0]].unsqueeze(1)
        Rel = enc_r[triples[:, 1]].view(-1, self.hidden_size, self.hidden_size)
        hR = torch.matmul(h, Rel).squeeze(1)
        return hR

    def get_reg_factor(self, triples: Tensor, enc_e, enc_r) -> Tensor:
        return enc_e[triples[:, 0]], enc_r[triples[:, 1]].view(-1, self.hidden_size, self.hidden_size), enc_e[triples[:, 2]] # enc_r = trans + rot
    
    
class RotE(DBModel):
    '''rotate + trans'''

    def __init__(self, args) -> None:
        super().__init__(args)
        self.emb_trans = nn.Embedding(self.n_rel * 2, self.hidden_size, device=args.device)
        self.bias = nn.Embedding(self.n_ent, 1, device=args.device)
        if args.init_scale > 0:
            self.emb_trans.weight.data = args.init_scale * torch.randn((self.n_rel * 2, self.hidden_size))
            self.bias.weight.data = args.init_scale * torch.randn((self.n_ent, 1))

        return
    
    def encode(self, triples: Tensor) -> Tensor:
        return (self.emb_ent.weight, self.bias.weight), (self.emb_rel.weight, self.emb_trans.weight) # here we need tensor, rather than Embedding
    
    def get_queries(self, triples, enc_e, enc_r) -> Tensor:
        rot_r, trans_r = enc_r # enc_r contains two part, rotation and translation
        rep, bias = enc_e

        h = rep[triples[:, 0]]
        rot_r = rot_r[triples[:, 1]]
        trans_r = trans_r[triples[:, 1]]
        hr = givens_rotation(rot_r, h) + trans_r
        hb = bias[triples[:, 0]]

        return hr, hb
    
    def get_candidates(self, triples, enc_e, eval_mode) -> Tensor:
        rep, bias = enc_e
        if not eval_mode:
            return rep[triples[:, 2]], bias[triples[:, 2]]
        else:
            return rep, bias # N_ent does not match BS, but this is not a problem, since these two kinds of setting are handled differently

    def score(self, v_queries, v_candidates, eval_mode=False, require_grad=True) -> Tensor:

        v_q, bias_q = v_queries
        v_c, bias_c = v_candidates

        # TODO continue here
        score = - euc_distance(v_q, v_c, eval_mode) + bias_q + (bias_c.t() if eval_mode else bias_c)
        
        return score
    
    def get_reg_factor(self, triples: Tensor, enc_e, enc_r) -> Tensor:
        return enc_e[0][triples[:, 0]], enc_e[1][triples[:, 1]], enc_r[0][triples[:, 1]], enc_r[1][triples[:, 1]], enc_e[0][triples[:, 2]], enc_e[1][triples[:, 2]],  # enc_r = trans + rot
    
            


        