from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from numpy import trapz

def get_metrics(path, boundary):
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

    # step, -1 if model cannot meet expectation
    step = -1 if max(y) < boundary else max(x)
    # The area enclosed by the completion rate curve and the left boundary
    area = (max(x) - min(x)) * max(y) - integrate.simps(y, x, dx=0.0001)
    return step, area, max(y)