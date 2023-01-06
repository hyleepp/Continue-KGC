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