import os
import random
from random import shuffle
import json

import pickle as pkl
import numpy as np
import torch
from torch import Tensor

import utils as dataset
from collections import defaultdict

 

def generate_scaffold(triples, n_ent, n_rel):
    """get the scaffold triples of a KG, scaffold means a graph containing all entities and relations

    Args:
        triples (list): list of triples
        n_ent (int): entity size
        n_rel (int): relation size

    Returns:
        scaffold_triples (list): list of scaffold triples
        scaffold_triples_set (set): set version of the above return value
    """

    visited_ent, visited_rel, scaffold_triples_set = [set() for _ in range(3)]
    scaffold_triples = []
    
    for triple in triples:
        h, r, t = triple
        if h not in visited_ent or t not in visited_ent or r not in visited_rel:
            visited_ent.add(h)
            visited_ent.add(t)
            visited_rel.add(r)

            scaffold_triples.append(triple)
            scaffold_triples_set.add((h, r, t))
    
    assert len(visited_ent) == n_ent and len(visited_rel) == n_rel, "Not any entity/relation is contained in the scaffold"

    return scaffold_triples, scaffold_triples_set

def merge_files(file_path, drop_first_line = True):
    """merge the original train / valid / test file into a unique file

    Args:
        file_path (str): the path of file
        drop_first_line (bool): whether to drop the first line, since some datasets this contains the number of data
    """
    splits = ['train', 'test', 'valid']
    total_triples = []
    for split in splits:
        with open(file_path + '/' + split + '.txt', 'r') as f:
            if drop_first_line:
                triples = f.readlines()[1:] # ! drop the first line
            print(len(triples))
            total_triples.extend(triples)
    
    # write triples
    with open(file_path + '/total' + '.txt', 'w+') as f:
        f.writelines(total_triples)
    
def write_triples(file_path, triples):
    """transform triples into index form and write it done

    Args:
        file_path (str): file path
        triples (list): e.g., (I, like, dog)
    """

    # get dict
    entity2id, relation2id = get_entity_and_relation_dict(triples)
    
    # transformed text to id
    transformed_triples = []
    for head, rel, tail in triples:
        head_id, rel_id, tail_id = entity2id[head], relation2id[rel], entity2id[tail]
        triple = [head_id, rel_id, tail_id]
        triple = ' '.join(list(map(lambda x: str(x), triple))) + '\n'
        transformed_triples.append(triple)


    print(transformed_triples[0])
    # write in a total file
    with open(file_path + '/total' + '.txt', 'w+') as f:
        f.writelines(transformed_triples)
    
    # transform entity2id and relation2id to strings
    entity2id_to_write = []
    for k, v in entity2id.items():
        entity2id_to_write.append(' '.join([k, str(v)]) + '\n')

    relation2id_to_write = []
    for k, v in relation2id.items():
        relation2id_to_write.append(' '.join([k, str(v)]) + '\n')

    # save entity2id and relation2id
    with open(file_path + '/entity2id.txt', 'w+') as f:
        f.writelines(entity2id_to_write)

    with open(file_path + '/relation2id.txt', 'w+') as f:
        f.writelines(relation2id_to_write)

    return

def switch_rel_and_tail(file_path, file_name):
    '''
    Used to switch the order of relation and tail, since some datasets store this in different order
    '''

    with open(os.path.join(file_path, file_name), "r") as f:
        triples = f.readlines()
    
    assert len(triples) != 0

    reordered_triples = []
    for triple in triples:
        h, r, t = triple.strip().split()
        reordered_triples.append(' '.join([h, t, r]) + '\n')
    
    with open(os.path.join(file_path, file_name), "w") as f:
        f.writelines(reordered_triples)
    
    return
    

def get_entity_and_relation_dict(triples: list):
    """get the entity2id and relation2id dicts

    Args:
        triples (list): (h, r, t)s

    Returns:
        tuple: entity2id and relation2id
    """
    
    entity2id = {}
    relation2id = {}
    ent_id = 0
    re_id = 0 
    
    print(triples[0])
    for h, r, t in triples:
        if h not in entity2id.keys():
            entity2id[h] = ent_id
            ent_id += 1
        if t not in entity2id.keys():
            entity2id[t] = ent_id
            ent_id += 1
        if r not in relation2id.keys():
            relation2id[r] = re_id
            re_id += 1

    
    return entity2id, relation2id


def merge_fb(file_path):
    """merge all the triples of fb and generate the correspoding entity and relation dicts 

    Args:
        file_path (str): the path of file
    """

    splits = ['train', 'test', 'valid']
    total_triples = []
    for split in splits:
        with open(file_path + '/' + split + '.txt', 'r') as f:
            triples = f.readlines()[1:] # drop the first line
            print(len(triples))
            total_triples.extend(triples)
    triples = [triple.split() for triple in total_triples]
    
    # get dict
    entity2id, relation2id = get_entity_and_relation_dict(triples)
    
    # transformed text to id
    transformed_triples = []
    for head, rel, tail in triples:
        head_id, rel_id, tail_id = entity2id[head], relation2id[rel], entity2id[tail]
        triple = [head_id, rel_id, tail_id]
        triple = ' '.join(list(map(lambda x: str(x), triple))) + '\n'
        transformed_triples.append(triple)


    print(transformed_triples[0])
    # write in a total file
    with open(file_path + '/total' + '.txt', 'w+') as f:
        f.writelines(transformed_triples)
    
    # transform entity2id and relation2id to strings
    entity2id_to_write = []
    for k, v in entity2id.items():
        entity2id_to_write.append(' '.join([k, str(v)]) + '\n')

    relation2id_to_write = []
    for k, v in relation2id.items():
        relation2id_to_write.append(' '.join([k, str(v)]) + '\n')

    # save entity2id and relation2id
    with open(file_path + '/entity2id.txt', 'w+') as f:
        f.writelines(entity2id_to_write)

    with open(file_path + '/relation2id.txt', 'w+') as f:
        f.writelines(relation2id_to_write)

def merge_wiki(file_path):

    splits = ['train_2015', 'valid_2015', 'test_2015']
    total_triples = []
    for split in splits:
        with open(file_path + '/' + split + '.csv', 'r') as f:
            triples = f.readlines()[1:] # drop the first line
            print(len(triples))
            total_triples.extend(triples)
    triples = [triple.strip().split(',') for triple in total_triples]
    print(triples[:10])

    write_triples(file_path, triples)

    return

def generate_active_learning_dataset(data_path, init_ratio=0.7, need_query_filter=False, random_seed=123,): 
    """generate the dataset and other things needed for active learning setting

    Args:
        data_path (str): path of data
        init_ratio (float, optional): how much portion data we know in the beginning. Defaults to 0.7.
        need_query_filter (bool, optional): whether or not generate the relation filter, this need wiki infomation, so only used to FB or wiki. Defaults to False.
        random_seed (int, optional): . Defaults to 123.
    """

    # generate a folder
    dataset.mkdir(data_path + "/active_learning" + f"/{init_ratio}") 

    # load total data
    with open(data_path + "/total.txt", 'r') as f:
        triples = f.readlines()
    
    # split the triples 
    triples = [triple.strip().split() for triple in triples]
    shuffle(triples)

    # generate the scaffold, i.e., ensure every entities appear in init_shape
    n_ent, n_rel = dataset.get_entity_and_relation_size(triples)

    n_init = int(len(triples) * init_ratio)

    assert n_ent <= n_init, "the initial size is too small"

    scaffold_triples, scaffold_triples_set = generate_scaffold(triples, n_ent, n_rel)

    # exclude the scaffold triples
    remain_triples = list(filter(lambda x: (x[0], x[1], x[2]) not in scaffold_triples_set, triples))

    # shuffle 
    if random_seed > 0:
        random.seed(random_seed)
    random.shuffle(remain_triples)

    # split the data into two parts
    unexplored_triples, init_beyond_scaffold_triples = remain_triples[:len(triples) - n_init], remain_triples[len(triples) - n_init:]
    init_triples = scaffold_triples + init_beyond_scaffold_triples

    assert abs(len(init_triples) - n_init) <= 1

    # save as pickle
    with open(os.path.join(data_path, "active_learning", str(init_ratio), "init_triples.pkl"), 'wb') as f:
        int_init_triples = dataset.triples_str_to_int(init_triples)
        ndarray_init_triples = np.asarray(int_init_triples).astype('int64')
        ndarray_init_triples = torch.from_numpy(ndarray_init_triples)
        pkl.dump(ndarray_init_triples, f)

    with open(os.path.join(data_path, "active_learning", str(init_ratio), "unexplored_triples.pkl"), 'wb') as f:
        int_unexplored_triples = dataset.triples_str_to_int(unexplored_triples)
        ndarray_unexplored_tripls = np.asarray(int_unexplored_triples).astype('int64')
        ndarray_unexplored_tripls = torch.from_numpy(ndarray_unexplored_tripls)
        pkl.dump(ndarray_unexplored_tripls, f)

    if need_query_filter:
        id2class = generate_id2class(data_path)
        query_filter = generate_query_filter(os.path.join(data_path, 'active_learning', str(init_ratio)), init_triples, id2class, n_rel, rec=True)

    # transform to writable ones
    init_triples = dataset.triples_to_lines(init_triples)
    unexplored_triples = dataset.triples_to_lines(unexplored_triples)

    # write txt version
    with open(os.path.join(data_path, "active_learning", str(init_ratio), "init_triples.txt"), 'w') as f:
        f.writelines(init_triples)
    
    with open(os.path.join(data_path, "active_learning", str(init_ratio), "unexplored_triples.txt"), 'w') as f:
        f.writelines(unexplored_triples)
    
    # write some characteristics about the dataset in Json
    dataset_config = {
        "n_ent": n_ent,
        "n_rel": n_rel,
        "init_ratio": init_ratio,
        "random_seed": random_seed
    }

    with open(os.path.join(data_path, "active_learning", str(init_ratio), "dataset_config.json"), 'w+') as f:
        dataset_config = json.dumps(dataset_config)
        f.write(dataset_config)


    
    print("create activating setting datasets done successfully.") 

    return

def generate_id2class(path):
    """generate the idx2class dict (1: 'human', 2: "film")
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
    """
        
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

    path_idx2class = os.path.join(path, 'idx2class.pkl') 

    with open(path_idx2class, 'wb') as f:
        pkl.dump(idx2class, f)
    # also save json for human reading
    path_idx2class = os.path.join(path, 'idx2class.json')
    with open(path_idx2class, 'w') as f:
        json.dump(idx2class, f)

    return idx2class 


def generate_query_filter(path:str, triples:list, id2class: dict, n_rel: int, rec=False) -> dict:
    """generate the query filter based on entity class (like 'human').
    That contains the legal pattern appeared in triples, like ['human', 'like'],
    and helps to filter out the ridiculous combination like ['human', 'isLocated']

    Args:
        path: to load and save filter
        triples (list): [(h,r,t)] 
        id2class (dict): {1: 'human'} 
        n_rel (int): how many rel, since we may add rec
        rec (bool): add reciprocal relations or not

    Returns:
        dict: like {human: [like, isWife, plays...]} 
    """

    class_filter_relation = defaultdict(set)
    for h, r, t in triples:
        h, r, t = int(h), int(r), int(t)
        # there may have some entity miss the corresponding descriptions
        h_class, t_class = id2class.get(h), id2class.get(t)
        if h_class:
            class_filter_relation[h_class].add(r)
        if rec and t_class:
            class_filter_relation[t_class].add(r + n_rel)
    
    path_filter = os.path.join(path, 'relation_filter.pkl') 
    with open(path_filter, 'wb') as f:
        pkl.dump(class_filter_relation, f)
    
    return class_filter_relation
    


if __name__ == "__main__":
    # merge_fb('/home/ljy/continue-completing-cycle/data/FB15K') 
    # merge_files('/home/ljy/continue-completing-cycle/data_raw/WN18/original')
    # switch_rel_and_tail('/home/ljy/continue-completing-cycle/data/WN18', 'total.txt')
    generate_active_learning_dataset('data/FB15K', 0.3, True)
    # merge_wiki('data_raw/wikikg-v2')
