from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from numpy import trapz

def get_metrics(path):
    # path = 'confirmed_logs/QuatE_No_reg/events.out.tfevents.1678184630.autodl-container-f77d11adac-015069fd.2076.0'

    ea=event_accumulator.EventAccumulator(path) 
    ea.Reload()
    # print(ea.scalars.Keys())
    
    val_psnr=ea.scalars.Items('completion_ratio')
    # print(len(val_psnr))
    # print([(i.step,i.value) for i in val_psnr])

    x = [i.step for i in val_psnr]
    y = [i.value for i in val_psnr]

    area = (max(x) - min(x)) * max(y) - integrate.simps(y, x, dx=0.0001)
    # print((max(x) - min(x)) * max(y) - integrate.trapz(y, x, dx=0.0001))
    # plt.xlabel('x axis')
    # plt.ylabel('y axis')
    # plt.legend(loc=4) # 指定legend的位置,读者可以自己help它的用法
    # plt.title('polyfitting')
    # plt.savefig('images/savefig_example.png')
    # plt.show()

    return area