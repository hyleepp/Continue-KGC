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

        return 
   
   # TODO murge the following two function
    @abstractmethod
    def encode(self, triples: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """triples, one-hot vectors (h, r, t) -> embeddings, dense vectors

        Args:
            triples (Tensor): one hot vectors
        
        Returns:
            Tensor: the embeddings of all entities and relations in triples
            
        """
        pass

    def decode(self, triples, emb_e, emb_r, eval_mode=False) -> Tensor:
        """calculate the score based on embeddings 

        Args:
            triples(Tensor): (BS x 3)
            emb_e (Tensor): the embeddings of all entities 
            emb_r (Tensor): the embeddings of relations
            eval_mode(Bool): use eval mode, means each (h, r) pair matches all candidate entities 

        Returns:
            Tensor: the score of each triple
                    if eval mode is False, then return (BS x 1)
                    else return (BS x N_ent)
        """
        emb_queries = self.get_queries(triples, emb_e, emb_r)
        emb_candidates = self.get_candidates(triples, emb_e, emb_r, eval_mode)

        scores = self.score(emb_queries, emb_candidates, eval_mode)

        return scores
    
    @abstractmethod
    def score(self, emb_queries, emb_candidates, eval_mode=False) -> Tensor:
        """calculate the scores given the embeddings of queries and candidates

        Args:
            emb_queries (_type_): the embeddings of queries 
            emb_candidates (_type_): the embedding of candidates
            eval_mode (bool, optional): 1-1 or 1-n_ent. Defaults to False.

        Returns:
            Tensor: if eval mode is True return shape(BS x 1), else return shape(BS x N_ent)
        """
        pass


    @abstractmethod
    def get_all_embeddings(self) -> Tuple:
        """return embeddings of all entities and relations, which is used on evaluation 

        Returns:
            Tuple: (emb_e, emb_r)
        """
        pass

    @abstractmethod
    def get_queries(self, triples, emb_e, emb_r) -> Tensor:
        """give embeddings of queries (h,r), return a vector, like (h + r)

        Args:
            triples (_type_): the given (h, r, t) triples
            emb_e (_type_): the embeddings of all entities after encoding 
            emb_r (_type_): the embeddings of all relations after encoding 

        Returns:
            Tensor: results of f(h, r), like translation
        """
        pass

    @abstractmethod
    def get_candidates(self, triples, emb_e) -> Tensor:
        """give the tail entities for corresponding triples

        Args:
            triples (_type_): the given (h, r, t) triples
            emb_e (_type_): the embeddings of all entities

        Returns:
            Tensor: the embedding of tail entities
        """
        pass


    def get_reg(self, triples: Tensor) -> Tensor:
        """calculate the regularization terms

        Args:
            triples (Tensor): input triples.

        Returns:
            Tensor: the reg terms
        """
        return 0 # the default setting, without reg


    def forward(self, triples: Tensor, eval_mode=False) -> Tuple[Tensor, Tensor]:
        """forward function

        Args:
            triples (Tensor): input triples
            eval_mode (bool, optional): whether or not use eval mode. Defaults to False.

        Returns:
            scores: the forward scores
            reg: reg terms
        """

        # triples -> embedding (may contains an extra encoder like gnn)
        emb_h, emb_r = self.encode(triples)
        # embedding -> score
        scores = self.decode(triples, emb_h, emb_r, eval_mode)
        
        # get regularization terms
        reg = self.get_reg(triples)

        return scores, reg
    
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
            q = triples.clone()
            if m == "lhs":
                q[:, 0], q[:, 1], q[:, 2] = q[:, 2], q[:, 1] + self.n_rel, q[:, 0]
                # get ranking
                ranks = self.get_ranking(q, filter=[m], batch_size=batch_size)
                mean_ranks = torch.mean(ranks).item()
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



