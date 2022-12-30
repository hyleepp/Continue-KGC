'''A basic class that contains all datas'''
import os
import pickle as pkl
from utils import dataset_utils

'''Switch cases'''
SETTING2SPLITS = {
    'active_learning': ['init_triples', 'unexplored_triples']
}

class KGDataset(object):

    def __init__(self, data_path, debug, setting) -> None:

        print("Initializing KG Dataset.")
        
        # init params
        self.data_path = data_path
        self.debug = debug
        self.data = {}

        # get splits setting
        try:
            splits = SETTING2SPLITS[setting] 
        except ValueError:
            print(f"Current setting {setting} is not implemented, please choose from {list(SETTING2SPLITS.keys())}")
        
        # load data
        all_triples = []
        for split in splits:
            file_path = os.path.join(self.data_path, split + '.txt')
            with open(file_path, 'rb') as f:
                self.data[split] = pkl.load(f)
                all_triples.extend(self.data[split])

        n_ent, n_rel = dataset_utils.get_entity_and_relation_size(all_triples)
        del all_triples # TODO write in a more elegant way


        


    def get_example(self, split):
        pass

    def get_shape(self):
        """return the entity and relation size
        """
