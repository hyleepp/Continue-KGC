import argparse
import logging
import os
import sys
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

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

    '''Basic Setting'''
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

    '''Pretrain Part '''
    parser.add_argument(
        "--counter", type=int, default=10, help='how many evaluations for waiting another rise in results'
    )
    parser.add_argument(
        "--max_epochs", type=int, default=200, help='training epochs'
    )
    parser.add_argument(
        "--need_pretrain", type=bool, action="store true", help="need pretrain in the init split?"
    )
    parser.add_argument(
        "--pretrain_learning_rate", type=float, default=1e-3, help='learning rate for pretraining'
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.9, help="train / (train + valid)"
    )
    parser.add_argument(
        "--valid_period", type=int, default=5, help="how often test performance on valid set"
    )

    '''Incremental Part''' 
    parser.add_argument(
        "--incremental_learning_rate", type=float, default=1e-3, help='learning rate for incremental learning'
    )
    parser.add_argument(
        "--active_num", type=int, default=1000, help= "how many active labels for each epoch"
    )
    parser.add_argument(
        "--expected_completion_ratio", type=float, default=0.99, help= "when the completion can be treated as almost done"
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


def organize_args(args):
    '''organize args into a hierarchial style'''
    pass

def prepare_logger(save_dir:str) -> None:
    '''Prepare logger and add a stream handler'''
    
    # init logger
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%M-%d %H:%M:%S",
        filename=os.path.join(save_dir, "run.log")
    )

    # Sync logging info to stdout
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console) # add the StreamHandler to root Logger
    logging.info(f"Saving logs in {save_dir}")

    return

def save_config(args, save_dir) -> None:
    '''save the config in both logger and json'''

    for k, v in sorted(vars(args).items()):
        logging.info(f'{k} = {v}')
    with open(os.path.join(save_dir, "config.json"), "w") as fjson:
        json.dump(vars(args), fjson)

    return

def initialization(args):
    """initialize settings like logger and return dataset, optimizer and model

    Args:
        args (dict): arguments

    Returns:
        model, dataset, optimizer
    """
    
    save_dir = get_savedir(args.model, args.dataset)

    prepare_logger(save_dir)

    # tensor board
    tb_writer = SummaryWriter(save_dir, flush_secs=5)

    # create dataset
    dataset_path = os.path.join(os.environ['DATA_PATH'], args.dataset)
    dataset = KGDataset(dataset_path, args.setting, args.debug)

    # save configs 
    save_config(args, save_dir)

    

    # Load data, default for active learning
    logging.info(f"\t Loading {args.dataset} in {args.setting} setting, with shape {str(dataset.get_shape)}")

    # TODO add info
    # TODO ce_weight

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

    return dataset, model, optimizer
    

def active_learning_running(dataset, model, optimizer, expected_completion_ratio, need_pretrain=False) -> None:

    # Data Loading
    init_triples = dataset.get_example('init')
    unexplored_triples = dataset.get_example('unexplored')

    # Init Training
    early_stop_counter = 0
    best_mrr = None
    best_epoch = None
    # TODO add other cases 
    logging.info("\t Start Init Training.")


    if need_pretrain:
        # seprate the init set into training and eval set
        train_count = int(len(init_triples) * args.train_ratio)
        train_triples, valid_triples = init_triples[:train_count], init_triples[train_count:]

        # pretraining
        for step in range(args.max_epochs):

            # Train step
            model.train()
            train_loss = optimizer.epoch(train_triples, 'train')
            logging.info(f"\t Epoch {step} | average train loss: {train_loss:.4f}")

            # Valid step
            model.eval()
            valid_loss = optimizer.epoch(valid_triples, 'valid')
            logging.info(f"\t Epoch {step} | average valid loss: {valid_loss:.4f}")

            # TODO write losses

            # Test on valid 
            if (step + 1) % args.valid == 0:
                pass

        # training with extra valid setting, use init to train

        # save the trained model
    else:
        # load model
        pass
    

    # continue active completion
    completion_ratio = len(init_triples) / (len(init_triples) + len(unexplored_triples))

    while completion_ratio < expected_completion_ratio:
        # prediction

        # incremental training

        # update completion ratio

        # save the current completion ratio
        pass




if __name__ == "__main__":
    args = prepare_parser()
    # args = organize_args(args) # TODO finish that 
    dataset, model, optimizer = initialization(args)
    