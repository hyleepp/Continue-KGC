'''A basic class that contains all dataset'''
import os
import pickle as pkl
from typing import Tuple
import json

from utils import dataset_utils

'''Switch cases'''
SETTING2SPLITS = {
    'active_learning': ['init', 'unexplored']
}

class KGDataset(object):

    def __init__(self, data_path, setting, debug=False) -> None:
        """initialize the KGDataset 

        Args:
            data_path (str): where the data is
            setting (str): which kind of setting we are going to use
            debug (bool, optional): whether or not use debug mode, means using a few data. Default to be False
        """

        print("Initializing KG Dataset.")
        
        # init params
        self.data_path = data_path
        self.debug = debug
        self.data = {}
        self.setting = setting
        self.n_ent = 0
        self.n_rel = 0

        # get splits setting
        try:
            splits = SETTING2SPLITS[setting] 
        except ValueError:
            print(f"Current setting {setting} is not implemented, please choose from {list(SETTING2SPLITS.keys())}")
        
        # load data
        for split in splits:
            file_path = os.path.join(self.data_path, setting, split + '_triples.pkl')
            with open(file_path, 'rb') as f:
                self.data[split] = pkl.load(f)
        
        # load the config file
        with open(os.path.join(self.data_path, setting, 'dataset_config.json'), 'r') as f:
            content = f.read()
            config = json.loads(content)

        self.n_ent, self.n_rel = config['n_ent'], config['n_rel']

        print("Load data successfully.")
        # TODO add the debug mode
        return 

    def get_example(self, split, use_reciprocal=True):
        """get examples in a split

        Args:
            split (str): string indicating the split to use
            use_reciprocal (bool, optional): where or not using reciprocal setting, i.e., turn tail prediction to head prediction. Defaults to True.

        Returns:
            examples: the data
        """
        # TODO use a switch style to implement other settings, here we only start with the active learning setting
        examples = []

        try:
            examples = self.data[split]
        except KeyError:
            print(f"Current setting \"{self.setting}\" does not have the split \"{split}\"")

        return examples
        

    def get_shape(self) -> Tuple[int, int]:
        """return the entity and relation size
        """
        return self.n_ent, self.n_rel

'''Testing functions'''
if __name__ == "__main__":
    dataset = KGDataset('data/WN18', 'active_learning')
    print(dataset.n_ent, dataset.n_rel)
    dataset.get_example('a')


