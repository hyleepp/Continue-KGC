from tensorboard.backend.event_processing import event_accumulator
import numpy as np
from scipy.integrate import simps, trapz

def get_metric(path, metric):
    # load log data
    ea=event_accumulator.EventAccumulator('confirmed_logs/QuatE_No_reg/events.out.tfevents.1678184630.autodl-container-f77d11adac-015069fd.2076.0') 
    ea.Reload()
    if metric == 'completion_ratio':
        val_psnr=ea.scalars.Items('completion_ratio')

        x, y = [i.step for i in val_psnr], [i.value for i in val_psnr]

        area = ((max(x) - min(x)) * max(y)) - simps(y, x, dx=0.0001)

        return area

    return None