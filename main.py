import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import ALL_MODELS 
from optimization.KGOptimizer import KGOptimizer
from optimization import Regularizer
from optimization.Regularizer import ALL_REGULARIZER
from utils.train import get_savedir, count_param
from dataset.KGDataset import KGDataset

''' Parser
'''
def prepare_parser():

    parser = argparse.ArgumentParser(
        description="setting for ACKGE"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="datasets"
    )
    parser.add_argument(
        "--model", type=str, required=True, choices=ALL_MODELS, help='model used in running'
    )
    parser.add_argument(
        "--regularizer", type=str, choices=ALL_REGULARIZER, default='F2', help= "choose a regularizer"
    )
    parser.add_argument(
        "--reg_weight", type=float, default=0, help='the weight of reg term'
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help='learning rate'
    )
    parser.add_argument(
        "--max_epochs", type=int, default=200, help='training epochs'
    )
    parser.add_argument(
        "--counter", type=int, default=10, help='how many evaluations for waiting another rise in results'
    )
    parser.add_argument(
        "--active_num", type=int, default=1000, help= "how many active labels for each epoch"
    )
    parser.add_argument(
        "--setting", type=str, choices=['active_learning'], help='which setting in KG'
    )
    parser.add_argument(
        "--debug", type=bool, action="store_true", help='whether or not debug the program'
    )
    parser.add_argument(
        "--device", type=str, choices=['cpu', 'cuda'], help="which device, cpu or cuda"
    )

    return parser.parse_args()


''' Preparation
'''

''' Train / Test
'''

def initialization(args):
    """initialize settings like logger and return dataset, optimizer and model

    Args:
        args (dict): arguments

    Returns:
        model, dataset, optimizer
    """
    
    save_dir = get_savedir(args.model, args.dataset)

    '''Logging'''

    # file logger

    # tensor board

    # create dataset
    dataset_path = os.path.join(os.environ['DATA_PATH'], args.dataset)
    dataset = KGDataset(dataset_path, args.setting, args.debug)

    # Load data, default for active learning
    # TODO add info
    # TODO add other cases
    init_triples = dataset.get_example('init')
    unexplored_triples = dataset.get_example('unexplored')

    # TODO ce_weight

    # save_config

    # create model
    model = getattr(models, args.model)(args)
    total_params = count_param(model)
    logging.info(f"Total number of parameters {total_params}")
    device = args.device
    model.to(device)

    # get optimizer
    regularizer = getattr(Regularizer, args.regularizer)(args.reg_weight)
    optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
    optimizer = KGOptimizer(model, optim_method, regularizer, args.neg_size, args.sta_scale, debug=args.debug)

    return (init_triples, unexplored_triples), model, optimizer
    

def active_learning_running()
    early_stop_counter = 0
    best_mrr = None
    best_epoch = None
    # TODO add other cases 
    logging.info("\t Start Init Training.")
    for step in range(args.max_epochs):

        model.train()
        

    #




if __name__ == "__main__":
    args = prepare_parser()
    dataset, model, optimizer = initialization(args)
    

    