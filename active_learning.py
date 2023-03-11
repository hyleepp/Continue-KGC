import logging
import os
import heapq

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import models
from optimization.KGOptimizer import KGOptimizer
from optimization import Regularizer
from utils.train import *
from dataset.KGDataset import KGDataset


class ActiveLearning(object):

    def __init__(self, args) -> None:

        print("Initializing")
        self.args = args  # todo write in a better way
        self.dataset, self.model, self.writer = self.initialization(args)
        self.device = self.model.device

        # in  phase 1 of pretrain, it will be changed to the real best epoch, which is used in phase 2 of pretrain
        self.best_epoch = args.max_epochs

        # Data Loading
        # TODO use dataloader
        # self.max_batch_for_inference = self.get_max_batch_for_inference() # this need to be done before load the following data, since it will occupy more memory
        self.max_batch_for_inference = args.max_batch_for_inference
        self.init_triples = self.dataset.get_triples('init', use_reciprocal=True).to(args.device)  # here we consider training is default to use reciprocal setting
        self.unexplored_triples = self.dataset.get_triples('unexplored', use_reciprocal=True).to(args.device)

        # TODO refactor
        triples_raw = self.dataset.get_triples('init')
        n_init = int(self.args.train_ratio * len(triples_raw))
        train_triples, valid_triples = triples_raw[:n_init], triples_raw[n_init:]
        train_triples = self.dataset.add_reciprocal(train_triples)  # only train needs rec
        self.train_triples = train_triples.to(args.device)
        self.valid_triples = valid_triples.to(args.device)
        # the filter setting in evaluation of mrr and hits
        self.entity_filters = get_seen_filters(self.train_triples, self.args.n_rel)
        del triples_raw, train_triples, valid_triples

        # used in incremental learning
        self.previous_true = self.init_triples  # rename for clarity
        self.previous_false = None  # verified to be false, rather than negative samples

        # use a hash function to help determine whether a new triples has appeared
        self.previous_true_set = tensor2set(self.init_triples)
        self.previous_false_set = set()

        self.unexplored_triples_set = tensor2set(self.unexplored_triples)

        self.new_true_list = []
        self.new_false_list = []

        print("Initialization finished")
        return

    def initialization(self, args):
        """initialize logger, dataset, writer, then return dataset, model and writer

        Args:
            args (dict): arguments

        Returns:
            model, dataset, optimizer
        """

        save_dir = get_savedir(args.model, args.dataset, args.gcn_type)
        args.save_dir = save_dir  # which will be used further

        prepare_logger(save_dir)

        # tensor board
        writer = SummaryWriter(save_dir, flush_secs=5)

        # create dataset
        dataset_path = os.path.join(os.environ['DATA_PATH'], args.dataset)
        dataset = KGDataset(dataset_path, args.setting,
                            args.init_ratio, args.debug)
        args.n_ent, args.n_rel = dataset.get_shape()  # add shape to args

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

    def get_validation_metric(self, valid_triples, total_graph) -> float:

        valid_metrics = self.model.calculate_metrics(valid_triples, total_graph, self.entity_filters)
        valid_metrics = avg_both(*valid_metrics)

        logging.info(f"MRR: {valid_metrics['MRR']:.3f}, Hits@1: {valid_metrics['hits@{1,3,10}'][0]:.3f}, Hits@3: {valid_metrics['hits@{1,3,10}'][1]:.3f}, Hits@10: {valid_metrics['hits@{1,3,10}'][2]:.3f} ")

        valid_mrr = valid_metrics['MRR']

        return valid_mrr

    def init_optimizer(self, reset_model=True):
        """init an optimizer

        Args:
            reset_model (bool, optional): renew a model and learning from scratch. Defaults to False.

        Returns:
            optimizer: _description_
        """

        if reset_model:
            self.model = getattr(models, self.args.model)(self.args).to(self.device)
        regularizer = getattr(Regularizer, self.args.regularizer)(self.args.reg_weight)
        optim_method = getattr(torch.optim, self.args.optimizer)(self.model.parameters(), lr=self.args.pretrain_learning_rate)
        optimizer = KGOptimizer(self.model, optim_method, regularizer, self.args.batch_size,
                                self.args.neg_size, self.args.sta_scale, debug=self.args.debug)
        self.model.train()

        return optimizer

    def pretrain(self) -> None:
        """get a pretrained model. if specify a trained one, load it, otherwise train one.
        """

        if not self.args.pretrained_model_id:

            logging.info("\t Do not specific a pretraiend model, then training from scratch.")

            # phase 1
            if not self.args.skip_phase_1:
                logging.info("\t Start pretraining phase 1: on training split.")
                self.pretrain_phase('train and valid')
                logging.info("\t Pretrain phase 1 optimization finished, get the best training epoch")
            else:
                logging.info("\t Skip the first stage")

            # phase 2
            logging.info("\t Start the pretrain phase 2: use all known data")
            self.pretrain_phase('united')
            logging.info("\t Pretrain phase 2 optimization finished.")

            # save model
            logging.info(f"\t Saving model in {self.args.save_dir}.")
            torch.save(self.model.cpu().state_dict(),os.path.join(self.args.save_dir, "model.pt"))
            self.model.cuda()
        else:
            logging.info(f"\t Load pretrained model.")
            self.model.load_state_dict(torch.load(os.path.join(self.args.pretrained_model_id, "model.pt")))
            logging.info("\t Load model successfully.")

        return

    def pretrain_phase(self, phase: str) -> None:
        """the function used in both pretraining phase. It train a KGE model in the most common way. 
        the different of two stage is whether of not split the triples into training and valid.

        Args:
            phase (str): which phase:
                1. train and valid: split into train and valid, and get the best epochs
                2. united: use all known data to train the best epochs we get on phase 1.

        Raises:
            ValueError: the phase is wrong 
        """

        assert phase in ['train and valid', 'united'], 'wrong phase name'
        if phase == 'united' and self.best_epoch == 0:
            raise ValueError("the value of best epoch has some question")

        # the model in two phrase are different
        optimizer = self.init_optimizer(reset_model=True)

        # prepare triples
        if phase == 'train and valid':
            # seprate the init set into training and eval set
            train_triples, valid_triples = self.train_triples, self.valid_triples
        else:
            train_triples = self.init_triples

        best_mrr = 0
        
        # training
        for step in range(self.best_epoch):

            # Train step
            self.model.train()
            train_loss = optimizer.pretraining_epoch(train_triples, 'train')
            logging.info(f"\t Epoch {step} | average train loss: {train_loss:.4f}")
            self.writer.add_scalar('train_loss', train_loss, step)

            # Valid step
            if phase == 'train and valid':  # only used in phase 1
                self.model.eval()
                valid_loss = optimizer.pretraining_epoch(valid_triples, 'valid')
                logging.info(f"\t Epoch {step} | average valid loss: {valid_loss:.4f}")
                self.writer.add_scalar('valid_loss', valid_loss, step)

                # Test on valid
                if (step + 1) % self.args.valid_period == 0:

                    valid_mrr = self.get_validation_metric(valid_triples, train_triples)

                    if not best_mrr or valid_mrr > best_mrr:
                        best_mrr = valid_mrr
                        counter = 0
                        self.best_epoch = step
                        logging.info("Best results updated, save current model.")

                    else:
                        counter += 1
                        if counter == self.args.patient:
                            logging.info("\t Early stopping.")
                            break

        if phase == 'train and valid':
            del self.train_triples, self.valid_triples  # not be used any more

        return

    def get_max_batch_for_inference(self):
        # ! this function is abandoned, since we don't know the memory mechanism instead torch, a large batch may lead to a collapse unexpectedly 
        ''' get the max batch size when try to get candidates.
            since the following process involves a tensor with dynamic shape (remain_scores_idx),
            it is hard to choose the best number once for all.
            The result of this function is good enough

            although we could use a try, expect strategy, this strategy itself will slow the inference, and we think this is unworthy
        '''
        # todo handle the multi-card problem

        left, right = 0, 100000 # a number large enough

        # dichotomy, does not need to be very precise
        while left < right + 1 and (right - left) / right > 0.01:
            mid = (left + right) // 2
            torch.cuda.empty_cache()
            try:
                query = torch.ones((mid, 2)).type(torch.LongTensor).to(self.device)
                scores, _ = self.model(query, eval_mode=True, require_reg=False)
                remain_scores_idx = (scores > float('-inf')).nonzero() # the idx of possible scores
                left = mid
                del query, scores, remain_scores_idx
            except:
                right = mid

        assert left > 0, "the memory is not enough, please choose a lower hidden size or do sth else"
        logging.info(f'max_batch_for_inference is {left + 1}')

        return left + 1

    def get_total_candidates(self, ):
        focus_entities = torch.arange(self.model.n_ent).to(self.model.device)  # get all nodes # ! default, can be improved
        candidate_queries = []
        for node in focus_entities:
            class_name = self.id2class.get(node.item()) # the class of this entity, like trump -> human
            if class_name:
                candidate_relations = list(self.query_filter.get(class_name))
                candidate_relations = torch.tensor(candidate_relations, dtype=node.dtype).to(node.device)
            else:
                candidate_relations = torch.arange(self.model.n_rel * 2).to(node.device)  # use all relations
            entity_col = torch.ones_like(candidate_relations) * node.item()
            candidate_queries.append(torch.stack((entity_col, candidate_relations), dim=1)) # (num_pairs, 2)

        candidate_queries = torch.cat(candidate_queries, dim=0)
        return candidate_queries

    def get_candidate_for_verification(self, candidate_queries) -> list:
        """try to give each possible link (or some of them, since we may have a filter function),
        and return the top-k highest candidates for verification in next step. Here we use a heap to help sort. 

        Returns:
            list: heap of possible candidates, each elements contains two thins, (score, triple)
        """

        self.model.eval()
        with torch.no_grad():
            # inference while updating?
            # TODO keep two separate processes, and keep synchronization
            # 1. get possible nodes (indics)
            # 1.1 just simply all nodes, save this one as a baseline
            # focus_entities = torch.arange(self.model.n_ent).to(self.model.device)  # get all nodes # ! default, can be improved
            # [0, 1, 2, ...]
            # 1.2 get nodes via deviation ||\hat{e} - e||
            # 1.3 save as a queue or sth like this (merged with 1.2)

            # 2. propose a possible relations
            # filter, maybe not so meaningful, since the tensor may not be sparse enough
            # candidate_queries = []
            # for node in focus_entities:
            #     class_name = self.id2class.get(node.item()) # the class of this entity, like trump -> human
            #     if class_name:
            #         candidate_relations = list(self.query_filter.get(class_name))
            #         candidate_relations = torch.tensor(candidate_relations, dtype=node.dtype).to(node.device)
            #     else:
            #         candidate_relations = torch.arange(self.model.n_rel * 2).to(node.device)  # use all relations
            #     entity_col = torch.ones_like(candidate_relations) * node.item()
            #     candidate_queries.append(torch.stack((entity_col, candidate_relations), dim=1)) # (num_pairs, 2)

            # candidate_queries = torch.cat(candidate_queries, dim=0)

            # TODO also limits the possible tails

            # 3. inference and get the scores
            # use a prior queue, this is not algorithm related, so will not be highlighted in paper
            heap = [HeapNode((float("-inf"), None)) for _ in range(self.args.active_num)]

            # batch run
            # here a batch is set to be 1000 # TODO more flexible
            batch_begin, batch_size = 0, 1

            # build triples
            # simply loop # TODO -> a little bit parallel
            # the less the active num, the faster this process
            # TODO 完全异步维护数据，全是gpu单向向cpu输入数据，然后cpu维护一个堆，最后两者结束同步就ok了

            # try to achieve the maximum capacity in the following progress

            # todo 在这里变成一种二分的模式, right 就是max，然后初始化为1000000，left就是cur，然后不断削减right的上限
            # 进两步，退一步，这样更加自适应一点点

            # todo use a dataloader in here, since randomly we will get abetter threshold for after steps of getting entities
            with tqdm(total=len(candidate_queries), unit='ex') as bar:
                bar.set_description("Get candidate progress")
                while batch_begin < len(candidate_queries):
                    queries = candidate_queries[batch_begin: batch_begin + batch_size] # (bs, 2)
                    scores, _ = self.model(queries, eval_mode=True, require_reg=False) # scores and reg_factor
                    passed_pair_idx = (scores > heap[0].value).nonzero() # the idx of possible scores, the idx is [query_idx, candidate_idx]
                    # store to heap and screen out unqualified ones in parallel style

                    self.update_heap(scores, queries, passed_pair_idx, heap)

                    bar.update(batch_size)
                    bar.set_postfix(min_score=f'{heap[0].value:.3f}')

                    batch_begin += batch_size
                    # initially give a mini batch to set a filter bar and gradually grow to active_num, if we initially use a huge batch, the first loop will be very slow. this can be treated as a warm up
                    batch_size = min(2 * batch_size, self.max_batch_for_inference)

        assert heap[-1].value != float("-inf"), "we meet some problems"

        return heap
    
    def update_heap(self, scores, queries, pair_idx, heap) -> None:
        """update the current heap with triples that has a score greater than the top of the heap.

        Args:
            scores (_type_): the scores of all triples, scores[i, j] means the score of i-th query and j-th candidate entity (or t for simplicity)
            queries (_type_): [[h, r]] 
            pair_idx (_type_): [query_idx (h, r), candidate_idx (t)] 
            heap (_type_): the heap that scores the most promised triples
        """
        # TODO see if we let this run on cpu and we continue gpu processes
        cur_seen = set() # to avoid the case that both ori triple and its rec one all appears
        for i in range(len(pair_idx)):
            query_idx, t = pair_idx[i]
            score = scores[query_idx, t]
            if score > heap[0].value:
                triple = torch.stack((queries[query_idx][0], queries[query_idx][1], t)) # tensor([h, r, t])
                # if triple not in previous_true and triple not in previous_false: # avoid rise what we have predicted
                triple_tuple = tuple(triple.tolist())
                if triple_tuple not in self.previous_true_set and \
                        triple_tuple not in self.previous_false_set and \
                        triple_tuple not in cur_seen:  # todo find some better way to do so
                    # Remove the top element and rearrange it after adding new elements so that the smallest element is at the top of the heap again
                    heapq.heapreplace(heap, HeapNode((score.item(), triple)))
                    reciprocal_tuple = (triple_tuple[2], triple_tuple[1] + self.model.n_rel, triple_tuple[0])
                    cur_seen.add(reciprocal_tuple) # the reciprocal triples should also be filtered, they are equivalent in unexplored set

        del queries, scores, pair_idx 
        torch.cuda.empty_cache()
        
        return 

    def verification(self, heap) -> tuple:
        """try to verify the candidate proposed by model are true or false in real graph.
        In real world, this part should be done by human, yet we use the triples in unexplored set as an alternative solution

        Args:
            heap (list): the heap of (score, triple)

        Returns:
            true_count:  how many candidates are true.
            false_count: how many candidates are false.
            completion_ratio: current completion ratio.
        """

        # verification
        true_count, false_count = 0, 0
        for node in heap:
            # see if this triple in unexplored
            triple = node.triple
            triple_tuple = tuple(node.triple.tolist())
            # build rec triple
            rec_triple = triple.clone()
            rec_triple[0], rec_triple[1], rec_triple[2] = triple[2], (triple[1] + self.model.n_rel) % (self.model.n_rel * 2), triple[0]
            rec_triple_tuple = (triple_tuple[2], (triple_tuple[1] + self.model.n_rel) % (self.model.n_rel * 2), triple_tuple[0])

            if triple_tuple in self.unexplored_triples_set:
                true_count += 1
                self.new_true_list.append(triple)
                self.new_true_list.append(rec_triple)
                self.previous_true_set.add(triple_tuple)  # previous == seen
                self.previous_true_set.add(rec_triple_tuple)
                self.unexplored_triples_set.remove(triple_tuple)
                self.unexplored_triples_set.remove(rec_triple_tuple)

            else:
                false_count += 1
                self.new_false_list.append(triple)
                self.new_false_list.append(rec_triple)
                self.previous_false_set.add(triple_tuple)
                self.previous_false_set.add(rec_triple_tuple)

        # update completion ratio
        completion_ratio = (len(self.previous_true_set)) / \
            (len(self.init_triples) + len(self.unexplored_triples))

        return true_count, false_count, completion_ratio

    def report_current_state(self, step, true_count, false_count, completion_ratio) -> None:
        '''report current state after a step of verification'''

        s = "After " + str(step) + "'th step of verification."
        print(f"{DOTS} {s:<50} {DOTS}")
        s = "Pred True " + str(true_count) + \
            " Pred False " + str(false_count) + "."
        print(f"{DOTS} {s:<50} {DOTS}")
        s = "Current Completion Ratio is " + \
            str(round(completion_ratio, 3)) + "."
        print(f"{DOTS} {s:<50} {DOTS}")
        self.writer.add_scalar("completion_ratio", completion_ratio, step)
        self.writer.add_scalar('precision', true_count / (true_count + false_count), step)

        return

    def incremental_learning(self, step: int) -> None:
        """use the previous verified data (both true and false) to train the model incrementally. 
        It should be noticed that incremental is not strict, since some baselines are not.

        Args:
            step (int): the current step 
        """
        # incremental learning
        # TODO add different inference method, i.e. may only inference a few, since it is also hard to update all these kind of things
        logging.info(f"\t Start incremental learning at step {step}")

        new_true = torch.stack(self.new_true_list) if self.new_true_list else None
        new_false = torch.stack(self.new_false_list) if self.new_false_list else None

        # reset list
        self.new_true_list = []
        self.new_false_list = []

        # incremental training
        reset_model = self.args.incremental_learning_method == 'reset'
        optimizer = self.init_optimizer(reset_model)

        # training
        avg_incre_loss = 0
        for incre_step in range(self.args.incremental_learning_epoch):
            incre_loss = optimizer.incremental_epoch(
                self.previous_true, self.previous_false, new_true, new_false, self.args.incremental_learning_method, self.args)
            self.writer.add_scalar('incre_loss', incre_loss, incre_step)
            avg_incre_loss += (incre_loss - avg_incre_loss) / (incre_step + 1)

        logging.info(f"\t Step {step} | average incre loss: {avg_incre_loss:.4f}")
        logging.info("\t Incremental learning finished.")

        # TODO use a basic incremental method instead of these two naive setting (finetune, retrain)

        # TODO think: do we just need to focus on the difference between ce and negative samples?

        # all-together and only utilize the true examples

        # put the new triples to previous
        # todo here is ambiguous, write in a better way
        if new_true is not None:
            self.previous_true = torch.cat((self.previous_true, new_true), 0)
        if new_false is not None and self.previous_false != None:
            self.previous_false = torch.cat(
                (self.previous_false, new_false), 0)
        elif new_false is not None:
            self.previous_false = new_false

        return

    def active_learning_running(self) -> None:
        """The setting that gives a set of known data (init triples), and continue completing it while human in the loop.

        It the repeat the following procedure:
        1. Model predict some possible missing links
        2. Human verified it (Or GT in experiments)
        3. update the model
        """

        self.pretrain()

        # continue active completion
        completion_ratio = len(self.init_triples) / (len(self.init_triples) + len(self.unexplored_triples))  # unexplored does not have reciprocal relations
        logging.info(f"Continue completion: {completion_ratio:.4f} -> {self.args.expected_completion_ratio} Starts.")
        step = 0

        # TODO more general
        logging.info('\t generate relation filter')
        self.id2class = load_id2class(self.dataset.data_path) if self.args.dataset == "FB15K" else None
        self.query_filter = load_query_filter(
            os.path.join(self.dataset.data_path, self.args.setting, str(self.args.init_ratio))) if self.args.dataset == "FB15K" else None
        logging.info('\t relation filter generated successfully')

        total_candidate = self.get_total_candidates()
        while completion_ratio < self.args.expected_completion_ratio:
            step += 1
            candidates = self.get_candidate_for_verification(total_candidate)
            true_count, false_count, completion_ratio = self.verification(candidates)
            self.report_current_state(step, true_count, false_count, completion_ratio)

            if self.args.update_freq > 0 and step % self.args.update_freq == 0:  # < 0 means never update
                self.incremental_learning(step)
            if step == 500:
                break

        logging.info(f"\t Completion finished at step {step}.")

        return
