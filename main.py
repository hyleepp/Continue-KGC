import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from utils.train import get_savedir
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
        "--model", type=str, required=True, help='model used in running'
    )
    parser.add_argument(
        "--setting", type=str, choices=['active_learning'], help='which setting in KG'
    )
    parser.add_argument(
        "--debug", type=bool, action="store_true", help='whether or not debug the program'
    )

    return parser.parse_args()


''' Preparation
'''

''' Train / Test
'''

def train(args):
    
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
    total = 


    



if __name__ == "__main__":
    args = prepare_parser()

    