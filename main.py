import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

''' Parser
'''
parser = argparse.ArgumentParser(
    description="setting for ACKGE"
)
parser.add_argument(
    "--dataset", type=str, required=True, help="datasets"
)
parser.add_argument(
    "--model", type=str, required=True, help='model used in running'
)


''' Preparation
'''

''' Train / Test
'''

if __name__ == "__main__":
    pass