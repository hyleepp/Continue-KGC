from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
import os 
import pickle
import argparse
from numpy import trapz
import math

def get_files(dir: str):
    filenames = os.listdir(dir)
    return filenames


def get_metrics(path, init_ratio, ideal_step):
    # path = 'confirmed_logs/QuatE_No_reg/events.out.tfevents.1678184630.autodl-container-f77d11adac-015069fd.2076.0'
    """
    Args:
        path: event file path
        boundary: default 0.99 
    Returns:
        step, area, max_completion
    """
    ea=event_accumulator.EventAccumulator(path) 
    ea.Reload()
    # print(ea.scalars.Keys())
    
    val_psnr=ea.scalars.Items('completion_ratio')
    # print(len(val_psnr))
    # print([(i.step,i.value) for i in val_psnr])

    x = [i.step for i in val_psnr]
    y = [i.value for i in val_psnr]

    # The area enclosed by the completion rate curve and the left boundary
    actual = integrate.simps(y, x, dx=0.0001) - (max(x) - min(x)) * min(y)
    ideal = ((max(x) - ideal_step) + (max(x) - min(x))) * (1.0 - init_ratio) * 0.5
    moar = actual / ideal
    return moar, max(y)


def create_parser():
    
    parser = argparse.ArgumentParser(
        description="setting for evaluation from tensorboard logs"
    )
    parser.add_argument("--tensorboard_path", type=str, help='path of tensorboard record in log')
    parser.add_argument("--init_ratio", type=float, help='the init ratio of dataset')
    parser.add_argument("--ideal_step", type=int, help='completion steps in an ideal state')

    return parser.parse_args()

args = create_parser()
moar, cr = get_metrics(args.tensorboard_path, args.init_ratio, args.ideal_step)
print("MOAR: {}, CR: {}".format(moar, cr))