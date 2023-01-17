import argparse
import logging
import os
import sys
import json
import heapq

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import models
from models import ALL_MODELS 
from optimization.KGOptimizer import KGOptimizer
from optimization import Regularizer
from optimization.Regularizer import ALL_REGULARIZER
from utils.train import get_savedir, count_param, avg_both
from dataset.KGDataset import KGDataset

''' Parser
'''
def set_environ():
    '''set some environment configs'''

    os.environ['KGHOME'] = "./"
    os.environ['LOG_DIR'] = "logs"
    os.environ['DATA_PATH'] = 'data'

    return 

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
        "--incremental_learning_method", type=str, required=True, choices=['retrain', 'finetune'], help="the specific method used in incremental learning"
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
        "--debug", action="store_true", help='whether or not debug the program'
    )
    parser.add_argument(
        "--device", type=str, default='cuda', choices=['cpu', 'cuda'], help="which device, cpu or cuda"
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
    """initialize logger, dataset, writer, then return dataset, model and writer

    Args:
        args (dict): arguments

    Returns:
        model, dataset, optimizer
    """
    
    save_dir = get_savedir(args.model, args.dataset)
    args.save_dir = save_dir # which will be used further

    prepare_logger(save_dir)

    # tensor board
    writer = SummaryWriter(save_dir, flush_secs=5)

    # create dataset
    dataset_path = os.path.join(os.environ['DATA_PATH'], args.dataset)
    dataset = KGDataset(dataset_path, args.setting, args.debug)
    args.n_ent, args.n_rel = dataset.get_shape() # add shape to args

    # save configs 
    save_config(args, save_dir)

    # Load data, default for active learning
    logging.info(f"\t Loading {args.dataset} in {args.setting} setting, with shape {str(dataset.get_shape())}")

    # TODO ce_weight

    # create model
    model = getattr(models, args.model)(args)
    total_params = count_param(model)
    logging.info(f"Total number of parameters {total_params}")
    device = args.device
    model.to(device)

    return dataset, model, writer
    

def active_learning_running(args, dataset, model, writer) -> None:

    # Data Loading
    init_triples = dataset.get_triples('init', use_reciprocal=True) # here we consider training is default to use reciprocal setting
    unexplored_triples = dataset.get_triples('unexplored', use_reciprocal=False)

    # Init Training
    early_stop_counter = 0
    best_mrr = None
    best_epoch = None
    # TODO add other cases 

    if not args.pretrained_model_id: 
        logging.info("\t Do not specific a pretraiend model, then training from scratch.")
        # Get optimizer
        regularizer = getattr(Regularizer, args.regularizer)(args.reg_weight)
        optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.pretrain_learning_rate)
        optimizer = KGOptimizer(model, optim_method, regularizer, args.batch_size, args.neg_size, args.sta_scale, debug=args.debug)

        # seprate the init set into training and eval set
        train_count = int(len(init_triples) * args.train_ratio)
        train_triples, valid_triples = init_triples[:train_count], init_triples[train_count:]

        # TODO build filters
        filters = None

        # pretraining only on training_set, to make sure the performance of the current setting

        # may be jumped
        logging.info("\t Start pretraining phase 1: on training split.")
        for step in range(args.max_epochs):

            # Train step
            model.train()
            train_loss = optimizer.epoch(train_triples, 'train')
            logging.info(f"\t Epoch {step} | average train loss: {train_loss:.4f}")

            # Valid step
            model.eval()
            valid_loss = optimizer.epoch(valid_triples, 'valid')
            logging.info(f"\t Epoch {step} | average valid loss: {valid_loss:.4f}")

            # write losses 
            writer.add_scalar('train_loss', train_loss, step)
            writer.add_scalar('valid_loss', valid_loss, step)

            # Test on valid 
            if (step + 1) % args.valid_period == 0:

                valid_metrics = model.calculate_metrics(valid_triples, filters)
                valid_metrics = avg_both(*valid_metrics)

                logging.info(f"MRR: {valid_metrics['MRR']:.3f}, Hits@1: {valid_metrics['hits@{1,3,10}'][0]:.3f}, Hits@3: {valid_metrics['hits@{1,3,10}'][1]:.3f}, Hits@10: {valid_metrics['hits@{1,3,10}'][2]:.3f} ")

                valid_mrr = valid_metrics['MRR']
                if not best_mrr or valid_mrr > best_mrr:
                    best_mrr = valid_mrr
                    counter = 0
                    best_epoch = step
                    logging.info("Best results updated, save current model.")

                else:
                    counter += 1
                    if counter == args.patient:
                        logging.info("\t Early stopping.")
                        break
        logging.info("\t Pretrain phase 1 finished Optimization finished, get the best training epoch")

        logging.info("\t Start the pretrain phase 2: both train and valid data")

        optimizer.optimizer.param_group[0]['lr'] = args.pretrain_learning_rate # reset optimizer

        for step in range(best_epoch): 

            # Train step
            model.train()
            train_loss = optimizer.epoch(init_triples, 'train') # all data
            logging.info(f"\t Epoch {step} | average train loss: {train_loss:.4f}")

            # write losses 
            writer.add_scalar('train_loss', train_loss, step)
        
        logging.info("\t Pretrain phase 2 finished Optimization finished.")

        # save model
        logging.info(f"\t Saving model in {args.save_dir}")
        torch.save(model.cpu().state_dict(), os.path.join(args.save_dir, "model.pt"))
        model.cuda()
    else:
        logging.info(f"\t Load pretrained model")
        # load model
        model.load_state_dict(torch.load(os.join(args.pretrained_model_id, "model.pt")))
        logging.info("\t Load model successfully")

    '''Incremental Learning Part'''
    previous_true = init_triples # rename for clarity
    previous_false = None # verified to be false, rather than negative samples

    # continue active completion
    completion_ratio = len(init_triples) / (len(init_triples) + len(unexplored_triples))
    # TODO add nested tqdm bars 
    step = 0
    while completion_ratio < args.expected_completion_ratio:
        step += 1

        # prediction
        model.eval()
        with torch.no_grad():
            # inference while updating?
            # TODO keep two separate processes, and keep synchronization
            # 1. get possible nodes (indics)
            # 1.1 just simply all nodes, save this one as a baseline
            focus_nodes = torch.range(model.n_ent) # get all nodes
            
            # 1.2 get nodes via deviation
            # 1.3 save as a queue or sth like this (merged with 1.2)
            
            # 2. propose a possible relations
            # 2.1 naive setting, get all relations for each node
            focus_relations = torch.range(model.n_rel * 2).unsqueeze_(0).repeat(focus_nodes, 1) # (nodes, rel)

            # 3. inference and get the scores
            # use a prior queue, this is not algorithm related, so will not be highlighted in paper
            heap = (float("-inf"), None) * args.active_num
            
            # batch run
            # here a batch is set to be 1000 # TODO more flexible
            ent_begin = 0 # it could also be treated as the row id in focus_relations
            rel_begin = 0 

            # build triples
            # simply loop # TODO -> a little bit parallel
            # TODO to be tested
            while ent_begin < len(focus_nodes):
                while rel_begin < len(focus_relations):
                    h, r = focus_nodes[ent_begin], focus_relations[ent_begin, rel_begin]
                    query = torch.cat((h, r)).unsqueeze_(0) #  add one dim, this will be removed in batched version
                    scores = model.forward(query, eval_mode=True) # (BS x N_ent)

                    # store to heap
                    # screen out unqualified ones in parallel style
                    remain_scores_idx = (scores > heap[0][0]).nonzero() # the idx of possible scores 

                    # update heap
                    for idx in remain_scores_idx:
                        score = idx 
                        if score > heap[0][0]:
                            triple = torch.cat(h, r, idx[1])
                            heapq.replace(heap, (score, triple))
                    
                    rel_begin += 1
                ent_begin += 1
            

            # while ent_begin < len(focus_nodes):
            #     while ent_begin < len(focus_nodes):
            #         while rel_begin < len(focus_relations):
            #             h, r = focus_nodes(ent_begin)
            #             score, _ = model() 

        # get answer 
        # TODO try parallel style
        new_true = []
        new_false = []
        for _, triple in heap:
            
            # see if this triple in unexplored 
            if triple in unexplored_triples:
                new_true.append(triple)
            else:
                new_false.append(triple)
        
        # update completion ratio
        completion_ratio = len(previous_true + new_true)
        # TODO update tqdm

        # TODO add different inference method, i.e. may only inference a few, since it is also hard to update all these kind of things

        # incremental training

        # modifed the learning rate and other settings of optimizer
        optimizer.optimizer.learning_rate = args.incremental_learning_rate
        optimizer.optimizer.param_group[0]['lr'] = args.pretrain_learning_rate # reset optimizer

        model.train()
        optimizer.incremental_training(previous_true, previous_false, new_true, new_false, args.incremental_learning_method, args)  

        # TODO use a basic incremental method instead of these two naive setting


        # TODO think: do we just need to focus on the difference between ce and negative samples?

        # all-together and only utilize the true examples
        
        
        # put the new triples to previous
        previous_true = torch.cat((previous_true, new_true), 0)
        previous_false = torch.cat((previous_false, new_false), 0) if previous_false else new_false

        # save the current completion ratio
        writer.add_scalar("completion_ratio", completion_ratio, step)
        pass




if __name__ == "__main__":

    set_environ()

    args = prepare_parser()
    # args = organize_args(args) # TODO finish that 
    # TODO use not all args, but the specific part of args like args.base
    dataset, model, writer = initialization(args)

    # switch cases
    if args.setting == 'active_learning':
        active_learning_running(args, dataset, model, writer)
    