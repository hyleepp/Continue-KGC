'''optimization class'''
from typing import Tuple

from tqdm import tqdm

from numpy import ndarray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class KGOptimizer(nn.Module):

    def __init__(self, model, optimizer, regularizer, batch_size, neg_size=-1, sta_scale=1, debug=False, dyn_scale=False, weight=None, verbose=True) -> None:
        super().__init__()

        # check args 
        assert neg_size == -1 or neg_size > 1, f'The given argument negative sample size {neg_size} is not implemented, please choose -1 or a number greater than 0.'
        

        self.model = model 
        self.optimizer = optimizer
        self.regularizer = regularizer

        self.batch_size = batch_size
        self.neg_size = neg_size 
        self.sta_scale = sta_scale 
        self.dyn_scale = 1 if dyn_scale else nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean', weight=weight)
        
        self.debug = debug 
        self.verbose = verbose
        

    def lr_schedule(self) -> None:
        # TODO determine the params
        pass

    def get_negative_samples(self, triples) -> None:
        # TODO
        pass

    def calculate_loss(self, triples):
        '''Calculate the loss of the given triples'''

        
        if self.neg_size == -1:
            loss, reg = self.neg_sample_loss(triples)
        else:
            loss, reg = self.no_neg_sample_loss(triples)
        pass

    def neg_sample_loss(self, triples) -> Tuple[Tensor, Tensor]:
        '''The Loss based on negative sample'''
        
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
        '''The loss caluctae without negative sample, like CE'''

        predictions, reg = self.model(triples, eval_mode=True)
        truth = triples[:, 2]

        loss = self.loss_fn(predictions, truth) # TODO add other losses

        return loss, reg


    def epoch(self, triples: Tensor, split: str) -> Tensor:
        """run one epoch of training

        Args:
            triples (Tensor): training triples in shape (N_train x 3)
            split (str): on train or eval

        Returns:
            loss: results of of that batch
        """

        assert split in ['train', 'valid'], "split must be train or valid"

        shuffled_triples = triples[torch.randperm(triples.shape[0]), :] if split == 'train' else triples # it is unnecessary to shuffle in valid
        with torch.enable_grad() if split == 'train' else torch.no_grad():
            # TODO test if this one works
            with tqdm(total=shuffled_triples.shape[0], unit='ex', disable=not self.verbose) as bar:
                bar.set_description(f"{split}_loss")
                b_begin = 0
                total_loss = 0
                counter = 0

                while b_begin < shuffled_triples.shape[0]:
                    # get input batch
                    input_batch = shuffled_triples[b_begin: b_begin + self.batch_size].cuda()

                    # forward and backward
                    if split == 'train':
                        loss = self.calculate_loss(input_batch)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    elif split == 'valid':
                        loss = self.calculate_loss(input_batch)
                        total_loss += loss

                    # prepare for next batch
                    b_begin += self.batch_size

                    # update tqdm bar
                    total_loss += 1
                    counter += 1
                    bar.update(input_batch.shape[0])
                    bar.set_postfix(loss=f'{loss.item():.4f}')

            total_loss /= counter
        return total_loss

    def calculate_valid_loss(self, triples: Tensor):

        b_begin = 0
        loss = 0
        counter = 0

        with torch.no_grad():
            while b_begin < triples.shape[0]:
                input_batch = triples[b_begin, b_begin + self.batch_size].cuda()
                
                b_begin += self.batch_size