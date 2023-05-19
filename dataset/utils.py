
'''utils'''
import os

import numpy as np
from numpy import ndarray

def mkdir(path):

    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print(f"new folder {path} created successfully.")
    
    else:
        print(f"the folder {path} is already existed.")

def get_entity_and_relation_size(triples: list):

    n_ent, n_rel = 0, 0
    for h, r, t in triples:
        n_ent = max(n_ent, int(h)) 
        n_ent = max(n_ent, int(t))
        n_rel = max(n_rel, int(r))
    
    return n_ent + 1, n_rel + 1

def triple_to_line(triple):
    '''transform a [h ,r, t] in a line for write
    '''
    return ' '.join(triple) + '\n'
    

def line_to_triple(line):
    '''transform a 'h r t\n' to [h, r t]
    '''
    return line.strip().split()
    

def triples_to_lines(triples):
    """transform the triples to lines with 'h r t\n'

    Args:
        triples (list): [h, r, t]

    Returns:
        ret: ['h r t\n'], except the last item 
    """
    ret = list(map(triple_to_line, triples))
    ret[-1] = ret[-1][:-1] # remove the extra '\n' 

    return ret

def lines_to_triples(lines):
    """transform the lines to triples

    Args:
        lines (list): lines of 'h r t\n'

    Returns:
        ret: list of triples
    """

    ret = list(map(line_to_triple, lines))
    return ret
    
def triples_str_to_int(triples) -> list:
    """transform the triples to lines with 'h r t\n'

    Args:
        triples (list): [h, r, t]

    Returns:
        ret: ['h r t\n'], except the last item 
    """
    ret = []
    for h, r, t in triples:
         ret.append((int(h), int(r), int(t)))
    return ret 


