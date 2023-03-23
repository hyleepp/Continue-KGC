
from matplotlib import pyplot as plt
import pickle as pkl
import seaborn as sns 

# nums = [1, 10, 100, 1000, 10000, 100000]
# nums = [1, 10, 100, 1000]
exps = list(range(5, -1, -1))
# colors = sns.color_palette("autumn", n_colors=len(nums)).as_hex()
colors = sns.color_palette("rainbow", n_colors=len(exps)).as_hex()

data = []
for exp in exps:
    with open(f'draw_lines/active_num_ablation/{10 ** exp}.pkl', 'rb') as f:
        datum = pkl.load(f)
        time, ratio = [x[0] for x in datum], [x[1] for x in datum]
        data.append([time, ratio])

for i, line in enumerate(data):
    time, ratio = line
    plt.plot(time, ratio, color=colors[i], label=f'$10^{exps[i]}$')

plt.xlabel('Time(s)')
plt.ylabel('Progress Ratio')
# plt.xscale('log')
plt.legend(title='Verification Nums')
plt.savefig('active_num_ablation.pdf')