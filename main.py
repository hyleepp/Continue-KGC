import argparse
import os

from models import ALL_MODELS 
from optimization.Regularizer import ALL_REGULARIZER
from active_learning import ActiveLearning
from utils.train import set_environ

# torch.backends.cudnn.enable =True
# torch.backends.cudnn.benchmark = True



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
    parser.add_argument(
        "--hidden_size", type=int, default=200, help="hidden dimension of embedding models"
    )
    parser.add_argument(
        "--optimizer", type=str, default="Adam", choices=["Adam, Adagrad"], help="optimizer"
    )
    parser.add_argument(
        "--sta_scale", type=float, default=1, help="scale factor in loss function"
    )
    parser.add_argument(
        "--dyn_scale", action="store_true", help="whether or not add a learnable factor"
    )
    parser.add_argument(
        "--init_ratio", type=float, required=True, help="the initial ratio of the triples"
    )
    parser.add_argument(
        "--debug", action="store_true", help='whether or not debug the program'
    )
    parser.add_argument(
        "--device", type=str, default='cuda', choices=['cpu', 'cuda'], help="which device, cpu or cuda"
    )

    '''Pretrain Part '''
    parser.add_argument(
        "--counter", type=int, default=10, help='how many evaluations for waiting another rise in results'
    )
    parser.add_argument(
        "--max_epochs", type=int, default=200, help='training epochs'
    )
    parser.add_argument(
        "--pretrained_model_id", type=str, help="load the pretrained model, if not then need pretrain"
    )
    parser.add_argument(
        "--neg_size", type=int, default=-1, help="if -1, means not use negative sample, else means the size of negative sample"
    )
    parser.add_argument(
        "--pretrain_learning_rate", type=float, default=1e-3, help='learning rate for pretraining'
    )
    parser.add_argument(
        "--skip_phase_1", action="store_true", help="directly use both train and valid data to pretrain, skip the phase searching the early stopping epoch, and will directly use max_epochs as the training epochs while ignore early stopping"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.9, help="train / (train + valid)"
    )
    parser.add_argument(
        "--valid_period", type=int, default=5, help="how often test performance on valid set"
    )
    parser.add_argument(
        "--batch_size", type=int, default=500, help="batch size"
    )
    parser.add_argument(
        "--patient", type=int, default=10, help="how many evaluation before early stopping"
    )
    parser.add_argument(
        "--init_scale", type=float, default=1e-3, help="the init scale of embeddings"
    )

    '''Incremental Part''' 
    parser.add_argument(
        "--incremental_learning_method", type=str, required=True, choices=['retrain', 'finetune', 'reset'], help="the specific method used in incremental learning"
    )
    parser.add_argument(
        "--incremental_learning_epoch", type=int, default=10, help="the epoch of incremental learning"
    )
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
        "--setting", type=str, required=True, choices=['active_learning'], help='which setting in KG'
    )
    parser.add_argument(
        "--update_freq", type=int, default=1, help='how many step to do an incremental learning'
    )
    parser.add_argument(
        "--max_batch_for_inference", type=int, default=15000, help="the max batch when trying to get candidates"
    )
    parser.add_argument(
        "--diff_weight", type=float, default=1e-3, help="the weight on the difference between new embedding and cur embedding"
    )

    '''GCN config'''
    parser.add_argument("--gcn_type", type=str, default="None")
    parser.add_argument("--gcn_base", type=int, default=4)
    parser.add_argument("--gcn_dropout", type=float, default=0.2)
    return parser.parse_args()


def organize_args(args):
    '''organize args into a hierarchial style'''
    pass

if __name__ == "__main__":

    set_environ()
    args = prepare_parser()
    # args = organize_args(args) # TODO finish that 
    # TODO use not all args, but the specific part of args like args.base

    # switch cases
    if args.setting == 'active_learning':
        active_learning = ActiveLearning(args)
        active_learning.active_learning_running()
    