from matplotlib import pyplot as plt
import pickle as pkl
import seaborn as sns 
import matplotlib.ticker as ticker 
from matplotlib.colors import Normalize
import numpy as np

from scipy.optimize import curve_fit


def func(x, a, b, c):
    return a * np.power(x, b) + c

# nums = [1, 10, 100, 1000, 10000, 100000]
# nums = [1, 10, 100, 1000]
# fig, ax = plt.subplots(figsize=[10,10])
fig, ax = plt.subplots()
nums = list(range(101))
colors = sns.color_palette("Spectral", n_colors=len(nums)).as_hex()
lines = []
for num in nums:
    with open(f'draw_lines/progress_and_filter_rate/{num}.pkl', 'rb') as f:
        datum = pkl.load(f)
        prog, ratio = [x[0] for x in datum], [x[1] for x in datum]
        prog = [x / prog[-1] for x in prog]
        lines.append([prog, ratio])

lines.reverse()
for i, line in enumerate(lines):
    prog, ratio = line
    idx = len(nums) - i - 1
    plt.plot(prog, ratio,  color=colors[idx], label=f'${idx}$', linewidth=0.05, alpha=0.2)

# plt.colorbar(plt.cm.ScalarMappable(),label="Normalized Thrust [a.u.]")
fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=len(nums) - 1), cmap = sns.color_palette("Spectral", n_colors=len(nums), as_cmap=True)),
             ax=ax, label="$i$-th Step")
# plt.clim(0, len(nums) - 1)
plt.xlabel('Progress Rate')
plt.ylabel('Pass Rate')
# plt.ylim(0, 0.0001)
plt.yscale('log')
# plt.xscale('log')
# plt.legend(title='Max Batch Size')
plt.savefig('progress_num_pass_rate.pdf')

# data = []
# for exp in exps:
#     with open(f'draw_lines/active_num_ablation/{10 ** exp}.pkl', 'rb') as f:
#         datum = pkl.load(f)
#         time, ratio = [x[0] for x in datum], [x[1] for x in datum]
#         data.append([time, ratio])

