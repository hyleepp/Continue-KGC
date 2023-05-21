'''Base Model for distance or bilinear based models
'''

from numpy import ndarray
import torch 
import torch.nn as nn
from torch import Tensor 
import torch.nn.functional as F
import pickle
from copy import deepcopy
import sys

from .KGModel import KGModel
from .GNNModel import RGCN
from utils.calculation import euc_distance, givens_rotation, quaternion_rotation, edge_normalization

DISTANCE_MODELS = ['TransE', 'RotatE', 'RotE', 'QuatE', 'CP']
BILINEAR_MODELS = ['RESCAL', 'ComplEx', "UniBi_2"]
GCN_MODELS = ['RGCN', ]
EPSILON = 1e-15

class DBModel(KGModel):

    def __init__(self, args) -> None:

        super().__init__(args)
        self.args = args
        self.similarity_method = 'dist' # or dot 
        if args.gcn_type != "None":
            self.encoder = RGCN(args.n_ent, args.n_rel, args.gcn_base, args.hidden_size, args.gcn_dropout)
        return
    
    def encode(self, triples: Tensor):
        # triples: (bs, 3)
        # we do not need this part, it is used in GNN based models
        if self.args.gcn_type != 'None':
            src, rel, dst = triples.t()
            unique_entity, edge_idx = torch.unique(torch.cat((src, dst)), return_inverse=True)
            src, dst = edge_idx.view(2, -1)
            edge_index = torch.stack((src, dst)) # edge_index represents the index of entity src, dst in specific entities of this batch
            edge_type = rel
            edge_norm = edge_normalization(edge_type, edge_index, len(unique_entity), self.n_rel)
            x = self.encoder(self.emb_ent(unique_entity), edge_index, edge_type, edge_norm)
            new_emb = self.emb_ent.weight.clone()
            new_emb[unique_entity] = x
            return new_emb, self.emb_rel.weight

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
        self.similarity_method = 'dist'
        self.args = args

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

class CP(DBModel):

    def __init__(self, args) -> None:
        super().__init__(args)
        self.similarity_method = 'dot'
    
    def get_queries(self, triples, enc_e, enc_r) -> Tensor:
        h = enc_e[triples[:, 0]]
        r = enc_e[triples[:, 1]]
        return h * r
    
class ComplEx(DBModel):

    def __init__(self, args) -> None:
        super().__init__(args)
        self.args = args

    def get_queries(self, triples, enc_e, enc_r) -> Tensor:
        h = enc_e[triples[:, 0]]
        r = enc_r[triples[:, 1]]
        h_r, h_i = h[:, self.hidden_size // 2:], h[:, :self.hidden_size // 2]
        r_r, r_i = r[:, self.hidden_size // 2:], r[:, :self.hidden_size // 2]
        lhs_e = torch.cat([
            h_r * r_r - h_i * r_i,
            h_r * r_i + h_i * r_r
        ], 1)

        return lhs_e
    
    def score(self, lhs_e, rhs_e, eval_mode=False, require_grad=True):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e = lhs_e[:, :self.hidden_size // 2], lhs_e[:, self.hidden_size // 2:]
        rhs_e = rhs_e[:, :self.hidden_size // 2], rhs_e[:, self.hidden_size // 2:]
        if eval_mode:
            return lhs_e[0] @ rhs_e[0].transpose(0, 1) + lhs_e[1] @ rhs_e[1].transpose(0, 1)
        else:
            return torch.sum(
                lhs_e[0] * rhs_e[0] + lhs_e[1] * rhs_e[1],
                1, keepdim=True
            )
    
    def get_reg_factor(self, triples: Tensor, enc_e, enc_r) -> Tensor:

        h = enc_e[triples[:, 0]]
        h_r, h_i = h[:, self.hidden_size // 2:], h[:, :self.hidden_size // 2]
        r = enc_r[triples[:, 1]]
        r_r, r_i = r[:, self.hidden_size // 2:], r[:, :self.hidden_size // 2]
        t = enc_e[triples[:, 2]]
        t_r, t_i = t[:, self.hidden_size // 2:], t[:, :self.hidden_size // 2]

        head_f = torch.sqrt(h_r ** 2 + h_i ** 2 + EPSILON)
        rel_f = torch.sqrt(r_r ** 2 + r_i ** 2 + EPSILON)
        tail_f = torch.sqrt(t_r ** 2 + t_i ** 2 + EPSILON)

        return head_f, rel_f, tail_f

class QuatE(DBModel):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.args = args

    def get_queries(self, triples, enc_e, enc_r) -> Tensor:
        h = enc_e[triples[:, 0]]
        r = enc_r[triples[:, 1]]
        # p, q, u, v = r.chunk(dim=1, chunks=4)
        # s_a, x_a, y_a, z_a = h.chunk(dim=1, chunks=4)
        # denominator = torch.sqrt(p ** 2 + q ** 2 + u ** 2 + v ** 2)
        # p, q, u, v = p / denominator, q / denominator, u / denominator, v / denominator
        # _a = s_a * p - x_a * q - y_a * u - z_a * v
        # _b = s_a * q + p * x_a + y_a * v - u * z_a
        # _c = s_a * u + p * y_a + z_a * q - v * x_a
        # _d = s_a * v + p * z_a + x_a * u - q * y_a

        # lhs_e = torch.cat([_a, _b, _c, _d], dim=1)
        lhs_e = quaternion_rotation(r, h, right=True)
        return lhs_e

    def get_candidates(self, triples, enc_e, eval_mode) -> Tensor:
        """give the tail entities for corresponding triples

        Args:
            enc_e (_type_): embedding of all entities after encoding
            eval_mode (_type_): return the corresponding embeddings (BS x 1) or the embeddings of all candidate entities (N_ent x 1)

        Returns:
            Tensor: the embedding of tail or all entities
        """

        # TODO filter out the entities that do not envolved, which is essential for continue activate learning
        if not eval_mode:
            return enc_e[triples[:, 2]]
        else:
            return enc_e # N_ent does not match BS, but this is not a problem, since these two kinds of setting are handled differently
    
    def score(self, lhs_e, rhs_e, eval_mode=False, require_grad=True):
        a, b, c, d = lhs_e.chunk(dim=1, chunks=4)
        s_c, x_c, y_c, z_c = rhs_e.chunk(dim=1, chunks=4)

        if eval_mode:
            return sum([a.mm(s_c.transpose(0, 1)), b.mm(x_c.transpose(0, 1)), c.mm(y_c.transpose(0, 1)), d.mm(z_c.transpose(0, 1))])
        else:
            return torch.sum(a * s_c + b * x_c + c * y_c + d * z_c, dim=1, keepdim=True) # (BS, 1)

    def get_reg_factor(self, triples: Tensor, enc_e, enc_r) -> Tensor:
        h, r, t = enc_e[triples[:, 0]], enc_r[triples[:, 1]], enc_e[triples[:, 2]]
        return h, r, t
            
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

class UniBi_2(DBModel):

    def __init__(self, args) -> None:
        super().__init__(args)
        self.Rot_u = nn.Embedding(self.n_rel * 2, self.hidden_size, device=args.device) # introduce reciprocal relations
        self.Rot_v = nn.Embedding(self.n_rel * 2, self.hidden_size, device=args.device)
        self.similarity_method = 'dot'
        
        return
    
    def encode(self, triples: Tensor):
        return self.emb_ent.weight, (self.Rot_u.weight, self.emb_rel.weight, self.Rot_v.weight)

    def get_queries(self, triples, enc_e, enc_r) -> Tensor:
        Rot_u, Rel_s, Rot_v = enc_r

        h = enc_e[triples[:, 0]]
        h = F.normalize(h, p=2, dim=1)

        ru = Rot_u[triples[:, 1]]
        rs = Rel_s[triples[:, 1]]
        rv = Rot_v[triples[:, 1]]

        rs_max = torch.max(torch.abs(rs), dim=1, keepdim=True)[0]
        rs = rs / rs_max

        uh = givens_rotation(ru, h)
        suh = rs * uh
        lhs = givens_rotation(rv, suh)

        self.reg = [h, ru, rs, rv] # to avoid duplicated computation in reg

        return lhs
    
    def get_candidates(self, triples, enc_e, eval_mode) -> Tensor:
        if not eval_mode:
            return F.normalize(enc_e[triples[:, 2]], p=2, dim=1)
        else:
            return F.normalize(enc_e, p=2, dim=1) # N_ent does not match BS, but this is not a problem, since these two kinds of setting are handled differently
    
    def get_reg_factor(self, triples: Tensor, enc_e, enc_r) -> Tensor:
        t = enc_e[triples[:, 2]]
        t = F.normalize(t, p=2, dim=1)
        self.reg.append(t)
        return self.reg

    
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
    
    def encode(self, triples: Tensor):
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

        score = - euc_distance(v_q, v_c, eval_mode) + bias_q + (bias_c.t() if eval_mode else bias_c)
        
        return score
    
    def get_reg_factor(self, triples: Tensor, enc_e, enc_r) -> Tensor:
        return enc_e[0][triples[:, 0]], enc_e[1][triples[:, 1]], enc_r[0][triples[:, 1]], enc_r[1][triples[:, 1]], enc_e[0][triples[:, 2]], enc_e[1][triples[:, 2]],  # enc_r = trans + rot