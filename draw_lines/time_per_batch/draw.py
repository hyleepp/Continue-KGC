from matplotlib import pyplot as plt
import pickle as pkl
import math

with open('draw_lines/time_per_batch/without class filter filtered.pkl', 'rb') as f:
    times = pkl.load(f)
# times = [math.log10(x) for x in times]

fig, ax = plt.subplots(figsize=[6,6])
plt.plot(times, label='w/ Root Filter', linewidth=0.1)
plt.plot([times[0]] * len(times), '--', label='w/o Root Filter')
plt.legend(loc=[0.55, 0.2])
plt.xlabel('Step')
plt.yscale('log')
plt.ylabel('Time(s)')

axins = ax.inset_axes([0.4, 0.5, 0.4854, 0.3])
x1, x2, y1, y2 = 0, 20, 0.05, 10000
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_yscale('log')
axins.plot(times, 'o-', label='w/ Root Filter', linewidth=0.1, ms=3)
axins.plot([times[0]] * len(times), '--', zorder=1)

# axins.set_xticklabels([])
# axins.set_yticklabels([])
ax.indicate_inset_zoom(axins, edgecolor="grey")

plt.savefig('time_per_batch.pdf')