from matplotlib import pyplot as plt
import pickle as pkl
import math

with open('draw_lines/time_and_progress/no warm up.pkl', 'rb') as f:
    no_warm_data = pkl.load(f)

with open('draw_lines/time_and_progress/warm up.pkl', 'rb') as f:
    warm_data = pkl.load(f)


fig, ax = plt.subplots(figsize=[6,6])
no_warm_t, no_warm_p = [x[0] for x in no_warm_data], [x[1] for x in no_warm_data]
no_warm_t = [0] + no_warm_t
no_warm_p = [0] + no_warm_p
warm_t, warm_p = [x[0] for x in warm_data], [x[1] for x in warm_data]
warm_t = [0] + warm_t
warm_p = [0] + warm_p
plt.plot(no_warm_t, no_warm_p, label='w/o warm-up')
plt.plot(warm_t, warm_p, label='w/ warm-up')
# plt.legend(loc='lower right')
plt.ylabel('Progress Ratio')
plt.xscale('log')
plt.xlabel('Time(s)')
plt.legend()

axins = ax.inset_axes([0.48, 0.2, 0.4045, 0.25])
x1, x2, y1, y2 = 7590, 7605, -0.01, 0.05
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
# axins.set_yscale('log')
# axins.set_xscale('log')
axins.plot(no_warm_t, no_warm_p,'o-', ms=3)
# axins.plot([times[0]] * len(times), '--', )
ax.indicate_inset_zoom(axins, edgecolor="grey")

plt.savefig('time_and_progress.pdf')
