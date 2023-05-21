from matplotlib import pyplot as plt
import pickle as pkl
import seaborn as sns 
# sns.color_palette("Blues").
# colors = sns.color_palette("mako", n_colors=10).as_hex()

# colors = sns.color_palette("ch:start=.2,rot=-.3", n_colors=10).as_hex()

# 1000
# 2000
# 5000
# 10000
# nums = [1000, 2000, 5000, 10000]
nums = [2000, 4000, 6000, 8000, 10000, 12000, 14000]
# colors = sns.color_palette("autumn", n_colors=len(nums)).as_hex()
colors = sns.color_palette("rainbow", n_colors=len(nums)).as_hex()

data = []
for num in nums:
    with open(f'draw_lines/max_batch_ablation/{num}.pkl', 'rb') as f:
        datum = pkl.load(f)
        time, ratio = [x[0] for x in datum], [x[1] for x in datum]
        data.append([time, ratio])

for i, line in enumerate(data):
    time, ratio = line
    plt.plot(time, ratio, color=colors[i], label=f'{nums[i]}')

plt.xlabel('Time(s)')
plt.ylabel('Progress Ratio')
plt.legend(title=r'$b_m^{max}$')
plt.savefig('max_batch_ablation.pdf')