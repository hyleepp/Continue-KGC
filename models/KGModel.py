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

    @abstractmethod
    def get_all_embeddings(self) -> Tuple:
        """return embeddings of all entities and relations, which is used on evaluation 

        Returns:
            Tuple: (emb_e, emb_r)
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

        # do we need filter? or we just need sth else?
        # here we just neededk

        mean_rank = {}
        mean_reciprocal_rank = {}
        hits_at = {}

        for m in ["rhs", "lhs"]:
            q = triples.clone()
            if m == "lhs":
                q[:, 0], q[:, 1], q[:, 2] = q[:, 2], q[:, 1] + self.n_rel, q[:, 0]
                # get ranking

        pass


    def get_ranking(self, triples, filter, batch_size):

        ranks = torch.ones(len(triples)) # default to be 1

        with torch.no_grad():
            
            b_begin = 0
            emb_e, emb_r = self.encode() # distance/bilinear based just simply get the embeddings while gnn needs to forward on graph
            candidates = self.get_candidates(triples, emb_e, eval_mode=True) # get embeddings of all candidates
            # ? there may have unseen entities, which should be filtered

            while b_begin < len(triples):
                batch_triples = triples[b_begin: b_begin + batch_size].cuda()

                queries = self.get_queries(batch_triples, emb_e)
                true_candidates = self.get_candidates(batch_triples, emb_e, eval_model=False)

                scores = self.score_func

                scores = self.score_func(q, ) 
        
    def score(self, lhs, rhs)


        