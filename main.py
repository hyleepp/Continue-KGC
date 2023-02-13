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
from tqdm import tqdm

import models
from models import ALL_MODELS 
from optimization.KGOptimizer import KGOptimizer
from optimization import Regularizer
from optimization.Regularizer import ALL_REGULARIZER
from utils.train import get_savedir, count_param, avg_both, tensor2set, HeapNode, DOTS
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
    parser.add_argument(
        "--init_ratio", type=float, required=True, choices=[0.7, 0.8, 0.9], help="the initial ratio of the triples"
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
        "--jump_phase_1", action="store_true", help="directly use both train and valid data to pretrain, jump the phase searching the early stopping epoch, and will directly use max_epochs as the training epochs while ignore early stopping"
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
    dataset = KGDataset(dataset_path, args.setting, args.init_ratio, args.debug)
    args.n_ent, args.n_rel = dataset.get_shape() # add shape to args

    # save configs 
    save_config(args, save_dir)

    # Load data, default for active learning
    logging.info(f"\t Loading dataset {args.dataset} in {args.setting} setting, with shape {str(dataset.get_shape())}")

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
    init_triples = dataset.get_triples('init', use_reciprocal=True).to(args.device) # here we consider training is default to use reciprocal setting
    unexplored_triples = dataset.get_triples('unexplored', use_reciprocal=True).to(args.device)

    # Init Training
    early_stop_counter = 0
    best_mrr = None
    best_epoch = 0
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
        if not args.jump_phase_1:
            logging.info("\t Start pretraining phase 1: on training split.")
            for step in range(args.max_epochs):

                # Train step
                model.train()
                train_loss = optimizer.pretraining_epoch(train_triples, 'train')
                logging.info(f"\t Epoch {step} | average train loss: {train_loss:.4f}")

                # Valid step
                model.eval()
                valid_loss = optimizer.pretraining_epoch(valid_triples, 'valid')
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
            logging.info("\t Pretrain phase 1 optimization finished, get the best training epoch")
        else:
            best_epoch = args.max_epochs

        logging.info("\t Start the pretrain phase 2: use all known data")

        # * reset model and optimizer, train from scratch,
        # ? we may change to continue previous training results
        model = getattr(models, args.model)(args)
        model.to(args.device)
        optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.pretrain_learning_rate)
        optimizer = KGOptimizer(model, optim_method, regularizer, args.batch_size, args.neg_size, args.sta_scale, debug=args.debug)

        for step in range(best_epoch + 1): 

            # Train step
            model.train()
            train_loss = optimizer.pretraining_epoch(init_triples, 'train') # all data
            logging.info(f"\t Epoch {step} | average train loss: {train_loss:.4f}")

            # write losses 
            writer.add_scalar('train_loss', train_loss, step)
        
        logging.info("\t Pretrain phase 2 optimization finished.")

        # save model
        logging.info(f"\t Saving model in {args.save_dir}.")
        torch.save(model.cpu().state_dict(), os.path.join(args.save_dir, "model.pt"))
        model.cuda()
    else:
        logging.info(f"\t Load pretrained model.")
        # load model
        model.load_state_dict(torch.load(os.path.join(args.pretrained_model_id, "model.pt")))
        logging.info("\t Load model successfully.")

    logging.info("\t Incremental learning start.")
    # TODO modified this part
    optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.incremental_learning_rate)
    regularizer = getattr(Regularizer, args.regularizer)(args.reg_weight)
    optimizer = KGOptimizer(model, optim_method, regularizer, args.batch_size, args.neg_size, args.sta_scale, debug=args.debug)

    previous_true = init_triples # rename for clarity
    previous_false = None # verified to be false, rather than negative samples

    previous_true_set = tensor2set(init_triples) # use a hash function to help determine whether a new triples has appeared
    previous_false_set = set()

    unexplored_triples_set = tensor2set(unexplored_triples)

    new_true_list = []
    new_false_list = []

    # continue active completion
    completion_ratio = len(init_triples) / (len(init_triples) + len(unexplored_triples))  # unexplored does not have reciprocal relations
    step = 0

    while completion_ratio < args.expected_completion_ratio:
        # TODO put in a function
        
        step += 1

        # prediction
        model.eval()
        with torch.no_grad():
            # inference while updating?
            # TODO keep two separate processes, and keep synchronization
            # 1. get possible nodes (indics)
            # 1.1 just simply all nodes, save this one as a baseline
            focus_nodes = torch.arange(model.n_ent).to(model.device) # get all nodes # ! default
            
            # 1.2 get nodes via deviation
            # 1.3 save as a queue or sth like this (merged with 1.2)
            
            # 2. propose a possible relations
            # 2.1 naive setting, get all relations for each node
            focus_relations = torch.arange(model.n_rel * 2).unsqueeze_(0).repeat(len(focus_nodes), 1).to(model.device) # (nodes, rel)
            # 2.2 filter, maybe not so meaningful, since the tensor may not be sparse enough

            # TODO also limits the possible tails 

            # 3. inference and get the scores
            # use a prior queue, this is not algorithm related, so will not be highlighted in paper
            heap = [HeapNode((float("-inf"), None)) for _ in range(args.active_num)]
            
            # batch run
            # here a batch is set to be 1000 # TODO more flexible
            ent_begin = 0 # it could also be treated as the row id in focus_relations
            rel_begin = 0 

            # build triples
            # simply loop # TODO -> a little bit parallel
            # the less the active num, the faster this process
            # TODO 完全异步维护数据，全是gpu单向向cpu输入数据，然后cpu维护一个堆，最后两者结束同步就ok了
            batch_size = 1 # todo fix the the problem of single input
            with tqdm (total=len(focus_nodes) * len(focus_relations[0]), unit='ex') as bar:
                bar.set_description("Get candidate progress")
                cur_seen = set()
                while ent_begin < len(focus_nodes):
                    rel_begin = 0
                    while rel_begin < len(focus_relations[0]):
                        h, r = focus_nodes[ent_begin: ent_begin + batch_size], focus_relations[ent_begin: ent_begin + batch_size, rel_begin]
                        # h, r = focus_nodes[ent_begin], focus_relations[ent_begin, rel_begin]
                        query = torch.stack((h, r), dim=1) #  add one dim, this will be removed in batched version
                        scores, _ = model(query, eval_mode=True, require_reg=False) # (BS x N_ent) (h x t)

                        # store to heap and screen out unqualified ones in parallel style
                        remain_scores_idx = (scores > heap[0].value).nonzero() # the idx of possible scores 
                        remain_scores = scores[torch.where(scores > heap[0].value)]

                        # update heap
                        # TODO see if we let this run on cpu and we continue gpu processes 
                        for i in range(len(remain_scores_idx)):
                            lhs_idx, t = remain_scores_idx[i]
                            if remain_scores[i] > heap[0].value:
                                triple = torch.stack((h[lhs_idx], r[lhs_idx], t)) 
                                # if triple not in previous_true and triple not in previous_false: # avoid rise what we have predicted
                                triple_tuple = tuple(triple.tolist())
                                if triple_tuple not in previous_true_set and \
                                triple_tuple not in previous_false_set and \
                                triple_tuple not in cur_seen: # todo find some better way to do so
                                    heapq.heapreplace(heap, HeapNode((remain_scores[i], triple)))
                                    reciprocal_tuple = (triple_tuple[2], triple_tuple[1] + model.n_rel, triple_tuple[0])
                                    cur_seen.add(reciprocal_tuple) # the reciprocal triples should also be filtered
                        
                        rel_begin += 1
                        bar.update(len(h))
                        bar.set_postfix(min_score=f'{heap[0].value:.3f}')
                    ent_begin += batch_size
                    batch_size = min(10 * batch_size, args.active_num) # initially give a mini batch to set a filter bar and gradually grow to active_num, if we initially use a huge batch, the first loop will be very slow. this can be treated as a warm up

        # get answer 
        assert heap[-1].value != float("-inf"), "we meet some problems"

       

      
        # active verified are triples in unexplored part
        for node in heap:
            
            # see if this triple in unexplored 
            triple = node.triple
            triple_tuple = tuple(node.triple.tolist())
            rec_triple = triple.clone()
            rec_triple[0], rec_triple[1], rec_triple[2] = triple[2], (triple[1] + model.n_rel) % (model.n_rel * 2), triple[0]
            rec_triple_tuple = (triple_tuple[2], (triple_tuple[1] + model.n_rel) % (model.n_rel * 2), triple_tuple[0])

            if triple_tuple in unexplored_triples_set:
                new_true_list.append(triple)
                new_true_list.append(rec_triple)
                previous_true_set.add(triple_tuple) # previous == seen
                previous_true_set.add(rec_triple_tuple)
                unexplored_triples_set.remove(triple_tuple)
                unexplored_triples_set.remove(rec_triple_tuple)
                
            else:
                new_false_list.append(triple)
                new_false_list.append(rec_triple)
                previous_false_set.add(triple_tuple)
                previous_false_set.add(rec_triple_tuple)
            
        # update completion ratio
        completion_ratio = (len(previous_true_set)) / (len(init_triples) + len(unexplored_triples))

        # TODO add different inference method, i.e. may only inference a few, since it is also hard to update all these kind of things

        # TODO dict ?
        if (step + 1) % args.update_freq == 0:
            
            # merge previous triples into tensor
            # if new_true_list:
            #     new_true = torch.stack(new_true)
            # if new_false_list:
            #     new_false = torch.stack(new_false)
            
            new_true = torch.stack(new_true_list) if new_true_list else None
            new_false = torch.stack(new_false_list) if new_false_list else None

            # reset list
            new_true_list = []
            new_false_list = []

            # incremental training

            # modifed the learning rate and other settings of optimizer
            optimizer.optimizer.learning_rate = args.incremental_learning_rate
            optimizer.optimizer.param_groups[0]['lr'] = args.pretrain_learning_rate # reset optimizer
            model.train()

            # training 
            avg_incre_loss = 0
            for incre_step in range(args.incremental_learning_epoch):
                incre_loss = optimizer.incremental_epoch(previous_true, previous_false, new_true, new_false, args.incremental_learning_method, args)  
                writer.add_scalar('incre_loss', incre_loss, incre_step)
                avg_incre_loss += (incre_loss - avg_incre_loss) / (incre_step + 1)

            logging.info(f"\t Step {step} | average incre loss: {avg_incre_loss:.4f}")

            # TODO use a basic incremental method instead of these two naive setting (finetune, retrain)


            # TODO think: do we just need to focus on the difference between ce and negative samples?

            # all-together and only utilize the true examples

            # put the new triples to previous
            # todo here is ambiguous, write in a better way
            if len(new_true):
                previous_true = torch.cat((previous_true, new_true), 0) 
            if len(new_false) and previous_false != None: 
                previous_false = torch.cat((previous_false, new_false), 0) 
            elif len(new_false):
                previous_false = new_false
  
        # todo move to a function
        s = "After " + str(step) + "'th step of active learning."
        print(f"{DOTS} {s:<50} {DOTS}")
        s = "Pred True " + str(len(new_true)) + " Pred False " + str(len(new_false)) + "."
        print(f"{DOTS} {s:<50} {DOTS}")
        s = "Current Completion Ratio is " + str(round(completion_ratio, 3)) + "."
        print(f"{DOTS} {s:<50} {DOTS}")
        writer.add_scalar("completion_ratio", completion_ratio, step)

        
    print(f"Completion finished at step {step}.")
    return


if __name__ == "__main__":

    set_environ()

    args = prepare_parser()
    # args = organize_args(args) # TODO finish that 
    # TODO use not all args, but the specific part of args like args.base
    dataset, model, writer = initialization(args)

    # switch cases
    if args.setting == 'active_learning':
        active_learning_running(args, dataset, model, writer)
    