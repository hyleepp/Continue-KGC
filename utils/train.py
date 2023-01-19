import datetime 
import os

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
