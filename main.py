import argparse
import os
import sys
# mcdir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# # print(mcdir)
# sys.path.append(mcdir)

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.train import get_savedir

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

    return parser.parse_args()


''' Preparation
'''

''' Train / Test
'''

def train(agrs):
    
    save_dir = get_savedir()

if __name__ == "__main__":
    args = prepare_parser()

    