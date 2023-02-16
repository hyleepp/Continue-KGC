'''A basic class that contains all dataset'''
import os
import pickle as pkl
from typing import Tuple
import json

import torch


'''Switch cases'''
SETTING2SPLITS = {
    'active_learning': ['init', 'unexplored']
}

class KGDataset(object):

    def __init__(self, data_path, setting, init_ratio, debug=False) -> None:
        """initialize the KGDataset 

        Args:
            data_path (str): where the data is
            setting (str): which kind of setting we are going to use
            init_ratio (float): which division of KGs should be chosen
            debug (bool, optional): whether or not use debug mode, means using a few data. Default to be False
        """

        print("Initializing KG Dataset.")
        
        # init params
        self.data_path = data_path
        self.debug = debug
        self.data = {}
        self.setting = setting
        self.init_ratio = init_ratio
        self.n_ent = 0
        self.n_rel = 0 # ! here we do not explicitly multiply this one by 2, which counts reciprocal relations

        # get splits setting
        try:
            splits = SETTING2SPLITS[setting] 
        except ValueError:
            print(f"Current setting {setting} is not implemented, please choose from {list(SETTING2SPLITS.keys())}")
        
        # load data
        for split in splits:
            file_path = os.path.join(self.data_path, setting, str(init_ratio), split + '_triples.pkl')
            with open(file_path, 'rb') as f:
                self.data[split] = pkl.load(f)
        
        # load the config file
        with open(os.path.join(self.data_path, setting, str(init_ratio), 'dataset_config.json'), 'r') as f:
            content = f.read()
            config = json.loads(content)

        self.n_ent, self.n_rel = config['n_ent'], config['n_rel']

        print("Load data successfully.")
        # TODO add the debug mode
        return 

        
    def add_reciprocal(self, triples):
        """add reciprocal triple for each triple, i.e. add (t, r + n_rel, h) for each (h, r, t)

        Args:
            triples (tensor): triples 

        Returns:
            triples: concatenated triples
        """

        copy = triples.clone().detach()
        tmp = copy.clone().detach()[:, 0]
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] = copy[:, 1] + self.n_rel
        # copy[:, 0], copy[:, 1], copy[:, 2] = copy[:, 2], copy[:, 1] + self.n_rel, copy[:, 0]
        triples = torch.cat((triples, copy), dim=0)

        return triples

    def get_triples(self, split, use_reciprocal=False):
        """get triples in a split

        Args:
            split (str): string indicating the split to use
            use_reciprocal (bool, optional): where or not using reciprocal setting, i.e., turn tail prediction to head prediction. Defaults to True.

        Returns:
            examples: the data
        """
        # TODO use a switch style to implement other settings, here we only start with the active learning setting
        triples = []

        try:
            triples = self.data[split] # Tensor

            # add reciprocal relations
            if use_reciprocal:
                triples = self.add_reciprocal(triples)
            
        except KeyError:
            print(f"Current setting \"{self.setting}\" does not have the split \"{split}\"")

        return triples 
        

    def get_shape(self) -> Tuple[int, int]:
        """return the entity and relation size
        """
        return self.n_ent, self.n_rel

'''Testing functions'''
if __name__ == "__main__":
    dataset = KGDataset('data/WN18', 'active_learning')
    print(dataset.n_ent, dataset.n_rel)
    dataset.get_triples('a')


