import datetime 
import os
import json
from collections import defaultdict
import logging
from numpy import ndarray
from torch import Tensor
import pickle as pkl

DOTS = '********'

def get_savedir(model_name:str, dataset_name:str) -> str:
    '''get the save dir based on model and dataset names'''
    dt = datetime.datetime.now()
    date = dt.strftime("%m_%d")
    save_dir = os.path.join(
        os.environ["LOG_DIR"], date, dataset_name,
        model_name + dt.strftime('_%H_%M_%S')
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
        return self.value < other.value # triples are incompatible

def tensor2set(tensor) -> set:
    '''convert a (N x 3) tensor to a set of triples'''
    ret = set()

    for i in range(len(tensor)):
        ret.add((tensor[i][0].item(), tensor[i][1].item(), tensor[i][2].item()))
    
    return ret

def load_classes(path):
    """load the idx2class dict (1: 'human', 2: "film")
    the class information is get from the "is instance of" relation from wikidata. And we will
    use this class information to filter some ridiculous pairs of entity and relation (like [human, isLocated])

    ps:
    key means sth like /m/012qdp
    qid means id in wiki, like Q5
    idx means id in triples, int number, i-th entity

    Args:
        path (_type_): the path of data 

    Returns:
        idx2class: a dict that map each idx to its class like {1: 'human'}
        class_names: the values of idx2class
    """
    path_idx2class = os.path.join(path, 'idx2class.pkl') 

    if os.path.exists(path_idx2class):
        with open(path_idx2class, 'rb') as f:
            idx2class = pkl.load(f)
        return idx2class

        
    key2qid, key2idx, idx2class = [{} for _ in range(3)]

    with open(os.path.join(path, 'entity2wikidata.json'), 'r') as f:
        entity2wiki = json.load(f)
        for key, value in entity2wiki.items():
            key2qid[key] = value['wikidata_id']

    with open(os.path.join(path, 'entity2id.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            key, idx = line.strip().split()
            key2idx[key] = int(idx)

    with open(os.path.join(path, "entity_type.json"), 'r') as f:
        qid2class = json.load(f)

    for k, v in entity2wiki.items():
        wiki_id = v['wikidata_id']
        class_name = qid2class[wiki_id]['parent_entity_name']
        idx = key2idx[k]
        idx2class[idx] = class_name

    with open(path_idx2class, 'wb') as f:
        pkl.dump(idx2class, f)
    # also save json for human reading
    path_idx2class = os.path.join(path, 'idx2class.json')
    with open(path_idx2class, 'w') as f:
        json.dump(idx2class, f)


    
    
    return idx2class 

def generate_relation_filter(path:str, triples: Tensor, id2class: dict, n_rel: int, rec=False) -> dict:
    """generate the relation filter based on entity class (like 'human').
    That contains the legal pattern appeared in triples, like ['human', 'like'],
    and helps to filter out the ridiculous combination like ['human', 'isLocated']

    Args:
        path: to load and save filter
        triples (Tensor): [(h,r,t)] 
        id2class (dict): {1: 'human'} 
        n_rel (int): how many rel, since we may add rec
        rec (bool): add reciprocal relations or not

    Returns:
        dict: like {human: [like, isWife, plays...]} 
    """
    # generate the relation filter based on entity class (like human).
    path_filter = os.path.join(path, 'relation_filter.pkl') 

    if os.path.exists(path_filter):
        with open(path_filter, 'rb') as f:
            filter = pkl.load(f)
        return filter 

    class_filter_relation = defaultdict(set)
    for h, r, t in triples:
        h, r, t = h.item(), r.item(), t.item()
        # there may have some entity miss the corresponding descriptions
        h_class, t_class = id2class.get(h), id2class.get(t)
        if h_class:
            class_filter_relation[h_class].add(r)
        if rec and t_class:
            class_filter_relation[t_class].add(r + n_rel)
    
    with open(path_filter, 'wb') as f:
        pkl.dump(class_filter_relation, f)
    # also save json file
    path_filter = os.path.join(path, 'relation_filter.json') 
    with open(path_filter, 'w') as f:
        json.dump(class_filter_relation, f)

    
    return class_filter_relation
    

def show_the_cover_rate(unexplored_triples:list, init_filter:dict, idx2class:dict, n_rel:int) -> None:
    """show the cover rate of init filter on unexplored triples

    Args:
        unexplored_triples (list): [h, r, t]
        init_filter (dict): {class: [r1, r2....]}
        idx2class (dict): {1: 'human'}
        n_rel (int): the number of relations 
    """

    count_in_init = 0
    count_class_not_recorded = 0 # some entity miss the wiki information

    for triple in unexplored_triples:
        h, r, t = triple
        h_class, t_class = idx2class.get(h), idx2class.get(t)
        if h_class and t_class:
            if r in init_filter[h_class] or r + n_rel in init_filter[t_class]:
                count_in_init += 1
        else:
            count_class_not_recorded += 1 

    logging.info(f'init filter cover rate: {count_in_init / len(unexplored_triples)}')
    logging.info(f'init filter + ignore unrecorded cover rate: {(count_in_init + count_class_not_recorded) / len(unexplored_triples)}')

    return