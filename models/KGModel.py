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

    def __init__(self, args) -> None: # here we use args is because we need to implement various of models that need different params
        super().__init__()

        self.n_ent = args.n_ent
        self.n_rel = args.n_rel
        self.hidden_size = args.hidden_size

        # naive version of embeddings
        self.emb_ent = nn.Embedding(self.n_ent, self.hidden_size, device=args.device)
        self.emb_rel = nn.Embedding(self.n_rel * 2, self.hidden_size, device=args.device) # introduce reciprocal relations

        return 
   
   # TODO murge the following two function
    @abstractmethod
    def encode(self, triples: Tensor) -> Tuple[Tensor, Tensor]:
        """triples, one-hot vectors (h, r, t) -> embeddings, dense vectors

        Args:
            triples (Tensor): one hot vectors
        
        Returns:
            Tensor: the embeddings of **all** entities and relations in triples, since if we use a graph, we update all the graph (or a sub graoh)
                    In other word, return emb_e (N_ent, d), emb_R (N_rel, d)
        """
        pass

    def decode(self, triples, enc_e, enc_r, eval_mode=False) -> Tensor:
        """calculate the score based on embeddings 

        Args:
            triples(Tensor): (BS x 3)
            enc_e (Tensor): the embeddings of all entities after encoding
            enc_r (Tensor): the embeddings of relations after encoding
            eval_mode(Bool): use eval mode, means each (h, r) pair matches all candidate entities 

        Returns:
            Tensor: the score of each triple
                    if eval mode is False, then return (BS x 1)
                    else return (BS x N_ent)
        """
        v_queries = self.get_queries(triples, enc_e, enc_r)
        v_candidates = self.get_candidates(triples, enc_e, eval_mode)

        scores = self.score(v_queries, v_candidates, eval_mode)

        return scores
    
    @abstractmethod
    def score(self, v_queries, v_candidates, eval_mode=False) -> Tensor:
        """calculate the scores given the vectors of queries and candidates

        Args:
            emb_queries (_type_): the vectors of queries 
            emb_candidates (_type_): the vectors of candidates
            eval_mode (bool, optional): 1-1 or 1-n_ent. Defaults to False.

        Returns:
            Tensor: if eval mode is True return shape(BS x 1), else return shape(BS x N_ent)
        """
        pass

    @abstractmethod
    def get_queries(self, triples, enc_e, enc_r) -> Tensor:
        """give results of queries (h,r), return a vector, like (h + r), given the encoded embeddings of entities and relations

        Args:
            triples (_type_): the given (h, r, t) triples
            enc_e (_type_): the embeddings of all entities after encoding after encoding
            enc_r (_type_): the embeddings of all relations after encoding after encoding

        Returns:
            Tensor: results of f(h, r), like translation
        """
        pass

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
            return self.emb_ent.weight # N_ent does not match BS, but this is not a problem, since these two kinds of setting are handled differently



    def get_reg_factor(self, triples: Tensor, enc_e, enc_r) -> Tensor:
        """get the materials tha are needed for calculating the regularization terms

        Args:
            triples (Tensor): input triples.
            emb_e (_type_): the embeddings of all entities after encoding 
            emb_r (_type_): the embeddings of all relations after encoding 

        Returns:
            Tensor: the reg terms
        """
        return enc_e[triples[:, 0]], enc_r[triples[:, 1]], enc_e[triples[:, 2]] # the default setting, vectors of h, r, t


    def forward(self, triples: Tensor, eval_mode=False) -> Tuple[Tensor, Tensor]:
        """forward function

        Args:
            triples (Tensor): input triples
            eval_mode (bool, optional): whether or not use eval mode. Defaults to False.

        Returns:
            scores: the forward scores
            reg: reg terms
        """

        enc_e, enc_r = self.encode(triples) # it should be noticed that emb_e
        # embedding -> score
        scores = self.decode(triples, enc_e, enc_r, eval_mode)
        
        # get regularization terms
        reg_factor = self.get_reg_factor(triples, enc_e, enc_r)

        return scores, reg_factor
    
    def calculate_metrics(self, triples: Tensor, filters: dict, batch_size=500) -> Tuple:
        """calculate metrics given the triples 

        Args:
            triples (Tensor): input triples
            filters (dict): Dict with entities to skip per query for evaluation in the filtered setting
            batch_size (int, optional): literally. Defaults to 500.

        Returns:
            Tuple: mean ranks, mean reciprocal ranks, and hits@k (1, 3 10)
        """

        # TODO add filters

        mean_rank = {}
        mean_reciprocal_rank = {}
        hits_at = {}

        for m in ["rhs", "lhs"]:
            q = triples.clone().detach()
            if m == "lhs":
                tmp = q[:, 0]
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] = q[:, 1] = self.n_rel
                
                # get ranking
                ranks = self.get_ranking(q, filter=[m], batch_size=batch_size)
                mean_rank = torch.mean(ranks).item()
                mean_reciprocal_rank = torch.mean(1. / ranks).item()
                hits_at[m] = torch.FloatTensor((list(map(
                    lambda x: torch.mean((ranks <= x).float()).item(), [1, 3, 10]
                ))))

        return mean_rank, mean_reciprocal_rank, hits_at


    def get_ranking(self, triples, filter, batch_size) -> Tensor:
        """calculate the rank of the tail entity againist all candidate entities for each triple 

        Args:
            triples (_type_): triples
            filter (_type_): what is already true in KG, to get the filtered results
            batch_size (_type_): _description_

        Returns:
            _type_: (BS x 1) the rank of all triples
        """

        # TODO add filter related parts 

        ranks = torch.ones(len(triples)) # default to be 1

        with torch.no_grad():
            
            b_begin = 0
            emb_e, emb_r = self.encode() # distance/bilinear based just simply get the embeddings while gnn needs to forward on graph
            all_candidates = self.get_candidates(triples, emb_e, eval_mode=True) # get embeddings of all candidates
            # ? there may have unseen entities, which should be filtered

            while b_begin < len(triples):
                batch_triples = triples[b_begin: b_begin + batch_size].cuda()

                # get embeddings
                queries = self.get_queries(batch_triples, emb_e)
                target_candidates = self.get_candidates(batch_triples, emb_e, eval_model=False)
                
                all_scores = self.score(queries, target_candidates, eval_mode=True)
                target_scores = self.score(queries, all_candidates, eval_mode=False) 

                # TODO filter part
                # here we just simply filter out the target itself
                # give the true and filer a huge negative number to ignore it
                # TODO test this function
                for i, triples in enumerate(triples):
                    _, _, target = triples
                    all_scores[i, target] = -1e6 # 

                ranks[b_begin:b_begin + batch_size] += torch.sum(
                    (all_scores >= target_scores).float(), dim=1
                ).cpu() # 
                b_begin += batch_size
            
        return ranks



