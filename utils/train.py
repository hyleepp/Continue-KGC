import datetime
import os
import json
from collections import defaultdict
import logging
from numpy import ndarray
from torch import Tensor
import pickle as pkl

DOTS = '********'

def set_environ():
    '''set some environment configs'''

    os.environ['KGHOME'] = "./"
    os.environ['LOG_DIR'] = "logs"
    os.environ['DATA_PATH'] = 'data'

    return 

def get_savedir(model_name: str, dataset_name: str) -> str:
    '''get the save dir based on model and dataset names'''
    dt = datetime.datetime.now()
    date = dt.strftime("%m_%d")
    # if gcn_type == 'None':
    #     save_dir = os.path.join(
    #         os.environ["LOG_DIR"], date, dataset_name,
    #         model_name + '_' + regularizer + '_' + str(reg_weight) + '_' + incremental_learning_method
    #     )
    # else:
    #     save_dir = os.path.join(
    #         os.environ["LOG_DIR"], date, dataset_name,
    #         model_name + '_' + gcn_type + dt.strftime('_%H_%M_%S')
    #     )

    save_dir = os.path.join(
        os.environ["LOG_DIR"], date, dataset_name,
        model_name + '_' + dt.strftime('%H_%M_%S')
    )
    os. makedirs(save_dir)
    return save_dir


def count_param(model) -> int:
    '''Count the number of parameters of the given model'''
    total = 0

    for x in model.parameters():
        if x.requires_grad:
            res = 1
            for y in x.shape:
                res *= y
            total += res

    return total


def avg_both(mrs, mrrs, hits) -> dict:
    """average metrics in both direction

    Args:
        mrs (_type_): mean rank
        mrrs (_type_): mean reciprocal rank
        hits (_type_): hits @ 1, 3, 10

    Returns:
        dict: a dict that contains the averaged results
    """

    mr = (mrs['lhs'] + mrs['rhs']) / 2
    mrr = (mrrs['lhs'] + mrrs['rhs']) / 2
    hit = (hits['lhs'] + hits['rhs']) / 2

    return {'MR': mr, "MRR": mrr, 'hits@{1,3,10}': hit}


class HeapNode(object):
    '''help to use a heap to get top-k pairs of (score, triple) based on scores.
        since triples are incompatible, we shall ignore all of them
    '''

    def __init__(self, tuple) -> None:
        self.value, self.triple = tuple
        return

    def __lt__(self, other) -> bool:
        return self.value < other.value  # triples are incompatible


def tensor2set(tensor) -> set:
    '''convert a (N x 3) tensor to a set of triples'''
    ret = set()

    for i in range(len(tensor)):
        ret.add((tensor[i][0].item(), tensor[i]
                [1].item(), tensor[i][2].item()))

    return ret


def load_query_filter(path: str) -> dict:
    """load the query filter based on entity class (like human)
    That contains the legal pattern appeared in triples, like ['human', 'like'],
    and helps to filter out the ridiculous combination like ['human', 'isLocated']

    Args:
        path (str): path of the file

    Returns:
        dict: _description_
    """
    path_filter = os.path.join(path, 'relation_filter.pkl')

    with open(path_filter, 'rb') as f:
        filter = pkl.load(f)
    return filter


def load_id2class(path: str) -> dict:
    """load the idx2class dict (1: 'human', 2: "film")
    the class information is get from the "is instance of" relation from wikidata. And we will
    use this class information to filter some ridiculous pairs of entity and relation (like [human, isLocated])

    Args:
        path (_type_): the path of data 

    Returns:
        idx2class: a dict that map each idx to its class like {1: 'human'}
    """
    path_idx2class = os.path.join(path, 'idx2class.pkl')

    with open(path_idx2class, 'rb') as f:
        idx2class = pkl.load(f)
    return idx2class


def show_the_cover_rate(unexplored_triples: Tensor, init_filter: dict, idx2class: dict, n_rel: int) -> None:
    """show the cover rate of init filter on unexplored triples

    Args:
        unexplored_triples (Tensor): [h, r, t]
        init_filter (dict): {class: [r1, r2....]}
        idx2class (dict): {1: 'human'}
        n_rel (int): the number of relations 
    """

    count_in_init = 0
    count_class_not_recorded = 0  # some entity miss the wiki information

    for triple in unexplored_triples:
        h, r, t = triple
        h, r, t = h.item(), r.item(), t.item()
        h_class, t_class = idx2class.get(h), idx2class.get(t)
        if h_class and t_class:
            if r in init_filter[h_class] or r + n_rel in init_filter[t_class]:
                count_in_init += 1
        else:
            count_class_not_recorded += 1

    logging.info(
        f'init filter cover rate: {count_in_init / len(unexplored_triples)}')
    logging.info(
        f'init filter + ignore unrecorded cover rate: {(count_in_init + count_class_not_recorded) / len(unexplored_triples)}')

    return


def get_seen_filters(examples, n_relations):
    """Create seen filtering lists for evaluation.

    Args:
      examples: Numpy array of size n_examples x 3 containing KG triples
      n_relations: Int indicating the total number of relations in the KG

    Returns:
      lhs_final: Dictionary mapping queries (entity, relation) to filtered entities for left-hand-side prediction
      rhs_final: Dictionary mapping queries (entity, relation) to filtered entities for right-hand-side prediction
    """
    lhs_filters = defaultdict(set)
    rhs_filters = defaultdict(set)
    for example in examples:
        lhs, rel, rhs = list(map(lambda x: x.item(), example))
        rhs_filters[(lhs, rel)].add(rhs)
        lhs_filters[(rhs, rel + n_relations)].add(lhs)
    lhs_final = {}
    rhs_final = {}
    for k, v in lhs_filters.items():
        lhs_final[k] = sorted(list(v))
    for k, v in rhs_filters.items():
        rhs_final[k] = sorted(list(v))
    filters = {'lhs': lhs_final, "rhs": rhs_final}
    return filters


def prepare_logger(save_dir: str) -> None:
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
    # add the StreamHandler to root Logger
    logging.getLogger('').addHandler(console)
    logging.info(f"Saving logs in {save_dir}")

    return


def save_config(args, save_dir) -> None:
    '''save the config in both logger and json'''

    for k, v in sorted(vars(args).items()):
        logging.info(f'{k} = {v}')
    with open(os.path.join(save_dir, "config.json"), "w") as fjson:
        json.dump(vars(args), fjson)

    return
