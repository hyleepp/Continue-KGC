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
            loss, reg_factor = self.no_neg_sample_loss(triples)
        else:
            loss, reg_factor = self.neg_sample_loss(triples)

        # calculate reg 
        reg = self.regularizer(reg_factor) if reg_factor else 0 # the reg is not calculated during inference for saving time
        
        return loss + reg

    def neg_sample_loss(self, triples) -> Tuple[Tensor, Tensor]:
        '''The Loss based on negative sample'''
        
        # positive loss 
        positive_scores, reg_factor = self.model(triples)
        positive_scores *= self.sta_scale * self.dyn_scale
        positive_scores = F.logsigmoid(positive_scores)

        # negative loss 
        negative_samples = self.get_negative_samples(triples)
        negative_scores, _ = self.model(negative_samples)
        negative_scores *= self.sta_scale * self.dyn_scale
        negative_scores = F.logsigmoid(- negative_scores)

        # total loss 
        loss = - torch.cat([positive_scores, negative_scores], dim=0).mean()
        
        return loss, reg_factor
    
    def no_neg_sample_loss(self, triples) -> Tuple[Tensor, Tensor]:
        '''The loss caluctae without negative sample, like CE'''

        predictions, reg_factor = self.model(triples, eval_mode=True)
        truth = triples[:, 2]

        loss = self.loss_fn(predictions, truth) # TODO add other losses

        return loss, reg_factor
    
    def pretraining_epoch(self, triples, mode) -> Tensor:
        """pretraining one epoch

        Args:
            triples (_type_): given data
            mode (_type_): train or valid

        Returns:
            loss: 
        """

        assert mode in ['train', 'valid', 'inference'], "mode must be train or valid or inference"

        triples = triples[torch.randperm(triples.shape[0]), :] if mode == 'train' else triples # it is unnecessary to shuffle in valid

        with torch.enable_grad() if mode == 'train' else torch.no_grad(): # inference does not require backward
            with tqdm(total=triples.shape[0], unit='ex', disable=not self.verbose) as bar:
                bar.set_description(f"{mode}_loss") 
                loss = self.epoch(triples, bar, mode)
        
        return loss
            
    def incremental_epoch(self, previous_true, previous_false, cur_true, cur_false, method, args)->None:
        """incremental training one epoch

        Args:
            previous_true (_type_): previous true triples 
            previous_false (_type_): previous false triples
            cur_true (_type_): current true triples
            cur_false (_type_): current false triples
            method (_type_): the incremental learning method
            args (_type_): other args may used
        """

        # TODO replaced with a switch style
        # TODO consider how to handle verified false
        
        if method == 'retrain': 
            triples = torch.cat((previous_true, cur_true), 0)
        elif method == 'finetune':
            triples = cur_true
        
        if len(triples) == 0:
            return float('nan') # if there is not new triples, then do not need to calculate loss
        
        with tqdm(total=triples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f"Incremental training {method} loss")
            loss = self.epoch(triples, bar, mode='train')
        
        return loss

    def epoch(self, triples, bar, mode: str) -> Tensor:
        """the running core of an optimizer 

        Args:
            triples (Tensor): training triples in shape (N_train x 3)
            bar: the tqdm handler of a progress bar
            mode (str): on train or valid 

        Returns:
            total_loss: avg loss of the given batch
        """

    # TODO change the arguments, replaced with previous_true, prevoius_false, cur_true, cur_false

        b_begin = 0
        total_loss = 0
        counter = 0

        while b_begin < triples.shape[0]:
            # get input batch
            input_batch = triples[b_begin: b_begin + self.batch_size].cuda()

            # forward and backward (for train)
            if mode == 'train':
                loss = self.calculate_loss(input_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            elif mode == 'valid':
                loss = self.calculate_loss(input_batch)
                total_loss += loss

            # prepare for next batch
            b_begin += self.batch_size

            # update tqdm bar
            total_loss += loss
            counter += 1
            bar.update(input_batch.shape[0])
            bar.set_postfix(loss=f'{loss.item():.4f}')

        total_loss /= counter

        return total_loss
