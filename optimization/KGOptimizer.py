'''optimization class'''


from typing import Tuple

from numpy import ndarray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class KGOptimizer(nn.Module):

    def __init__(self, model, optimizer, regularizer, neg_size=-1, sta_scale=1, dyn_scale=False, weight=None) -> None:
        super().__init__()

        # check args 
        assert neg_size == -1 or neg_size > 1, f'The given argument negative sample size {neg_size} is not implemented, please choose -1 or a number greater than 0.'
        

        self.model = model 
        self.optimizer = optimizer
        self.regularizer = regularizer

        self.neg_size = neg_size 
        self.sta_scale = sta_scale 
        self.dyn_scale = 1 if dyn_scale else nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean', weight=weight)



    def lr_schedule(self) -> None:
        # TODO determine the params
        pass

    def get_negative_samples(self, triples) -> None:
        pass

    def calculate_loss(self, triples):

        
        if self.neg_size == -1:
            loss, reg = self.neg_sample_loss(triples)
        else:
            loss, reg = self.no_neg_sample_loss(triples)
        pass

    def neg_sample_loss(self, triples) -> Tuple[Tensor, Tensor]:
        
        # positive loss 
        positive_scores, reg = self.model(triples)
        positive_scores *= self.sta_scale * self.dyn_scale
        positive_scores = F.logsigmoid(positive_scores)

        # negative loss 
        negative_samples = self.get_negative_samples(triples)
        negative_scores, _ = self.model(triples)
        negative_scores *= self.sta_scale * self.dyn_scale
        negative_scores = F.logsigmoid(- negative_scores)

        # total loss 
        loss = - torch.cat([positive_scores, negative_scores], dim=0).mean()
        
        return loss, reg 
    
    def no_neg_sample_loss(self, triples) -> Tuple[Tensor, Tensor]:

        predictions, reg = self.model(triples, eval_mode=True)
        truth = triples[:, 2]

        loss = self.loss_fn(predictions, truth) # TODO add other losses

        return loss, reg

        
        





