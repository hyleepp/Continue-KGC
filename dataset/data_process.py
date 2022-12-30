import os
import random
import json

import pickle
import numpy as np

from utils import dataset_utils

 

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

def generate_active_learning_dataset(data_path, init_ratio=0.7, random_seed=123): 

    # generate a folder
    dataset_utils.mkdir(data_path + "/active_learning") 

    # load total data
    with open(data_path + "/total.txt", 'r') as f:
        triples = f.readlines()
    
    # split the triples 
    triples = [triple.strip().split() for triple in triples]

    # generate the scaffold, i.e., ensure every entities appear in init_shape
    n_ent, n_rel = dataset_utils.get_entity_and_relation_size(triples)

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
    with open(os.path.join(data_path, "active_learning", "init_triples.pkl"), 'wb') as f:
        int_init_triples = dataset_utils.triples_str_to_int(init_triples)
        ndarray_init_triples = np.asarray(int_init_triples).astype('int64')
        pickle.dump(ndarray_init_triples, f)

    with open(os.path.join(data_path, "active_learning", "unexplored_triples.pkl"), 'wb') as f:
        int_unexplored_triples = dataset_utils.triples_str_to_int(unexplored_triples)
        ndarray_unexplored_tripls = np.asarray(int_unexplored_triples).astype('int64')
        pickle.dump(ndarray_unexplored_tripls, f)

    # transform to writable ones
    init_triples = dataset_utils.triples_to_lines(init_triples)
    unexplored_triples = dataset_utils.triples_to_lines(unexplored_triples)

    # write txt version
    with open(os.path.join(data_path, "active_learning", "init_triples.txt"), 'w') as f:
        f.writelines(init_triples)
    
    with open(os.path.join(data_path, "active_learning", "unexplored_triples.txt"), 'w') as f:
        f.writelines(unexplored_triples)
    
    # write some characteristics about the dataset in Json
    dataset_config = {
        "n_ent": n_ent,
        "n_rel": n_rel,
        "init_ratio": init_ratio,
        "random_seed": random_seed
    }

    with open(os.path.join(data_path, "active_learning", "dataset_config.json"), 'w+') as f:
        dataset_config = json.dumps(dataset_config)
        f.write(dataset_config)
    
    print("create activating setting datasets done successfully.") 

    return


if __name__ == "__main__":
    # merge_fb('/home/ljy/continue-completing-cycle/data/FB15K') 
    # merge_files('/home/ljy/continue-completing-cycle/data_raw/WN18/original')
    # switch_rel_and_tail('/home/ljy/continue-completing-cycle/data/WN18', 'total.txt')
    generate_active_learning_dataset('data/FB15K')