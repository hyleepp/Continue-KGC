# import matplotlib.pyplot as plt

# import matplotlib.ticker as mtick

# # FB
# mrr_quate = [0.546, 0.548, 0.550, 0.549, 0.546, 0.546, 0.548, 0.547]
# moar_quate = [0.637, 0.575, 0.553, 0.503, 0.629, 0.620, 0.597, 0.576] 

# mrr_rotate = [0.543, 0.543, 0.545, 0.544, 0.543, 0.544, 0.544, 0.543]
# moar_rotate = [0.632, 0.614, 0.607, 0.571, 0.597, 0.599, 0.607, 0.614]

# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))

# ax2 = ax1.twinx()
# x = [i for i in range(len(mrr_quate))]
# line1 = ax1.plot(x, moar_quate, label='MOAR QuatE')
# line2 = ax1.plot(x, moar_rotate, label='MOAR RotatE')
# ax1.set_ylabel('MOAR')
# ax1.set_ylim(0.4, 0.7)

# line3 = ax2.plot(x, mrr_quate, label='MRR QuatE', linestyle='--')
# line4 = ax2.plot(x, mrr_rotate, label='MRR RotatE', linestyle='--')
# ax2.set_ylabel('MRR')
# ax2.set_ylim(0.54, 0.565)

# ticks = ['DURA_0.001', 'DURA_0.003', 'DURA_0.005', 'DURA_0.01', 'F2_0.001', 'F2_0.003', 'F2_0.005', 'F2_0.01']
# lines = line1 + line2 + line3 + line4
# labs = [l.get_label() for l in lines]
# ax1.legend(lines, labs)

# plt.xticks(x, ticks)
# plt.xticks(rotation=45)
# plt.savefig('Regularization.png')

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from matplotlib.pyplot import MultipleLocator

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# FB

mrr_quate = [0.546, 0.546, 0.548, 0.547]
moar_quate = [0.637, 0.628, 0.605, 0.583] 

mrr_rotate = [0.543, 0.544, 0.544, 0.543]
moar_rotate = [0.597, 0.599, 0.607, 0.614]

# mrr_quate = [0.594, 0.596, 0.597, 0.596]
# moar_quate = [0.757, 0.759, 0.757, 0.755] 

# mrr_rotate = [0.591, 0.593, 0.593, 0.593]
# moar_rotate = [0.763, 0.764, 0.768, 0.766]

fig, ax1 = plt.subplots()
ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))

ax2 = ax1.twinx()
ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))

x = [i for i in range(len(mrr_quate))]
line1 = ax1.plot(x, moar_quate, label='MOAR QuatE')
line2 = ax1.plot(x, moar_rotate, label='MOAR RotatE')
ax1.set_ylabel('MOAR')
ax1.set_ylim(0.40, 0.70)
ax1.set_xlabel('Weight')

line3 = ax2.plot(x, mrr_quate, label='MRR QuatE', linestyle='--')
line4 = ax2.plot(x, mrr_rotate, label='MRR RotatE', linestyle='--')
ax2.set_ylabel('MRR')
y_major_locator=MultipleLocator(0.005)

ax2.set_ylim(0.540, 0.565)
ax2.yaxis.set_major_locator(y_major_locator)

# ticks = ['DURA_0.001', 'DURA_0.003', 'DURA_0.005', 'DURA_0.01', 'F2_0.001', 'F2_0.003', 'F2_0.005', 'F2_0.01']
ticks = ['0.001', '0.003', '0.005', '0.01']
lines = line1 + line2 + line3 + line4
labs = [l.get_label() for l in lines]
ax1.legend(lines, labs)

ax1.set_xticks(x)
ax1.set_xticklabels(ticks)

# Adjust the position of the x-axis label
# ax1.xaxis.set_label_coords(0.5, -0.15)

# plot scatter
scatter_x = [0, 3]
scatter_y = [0.637, 0.614]
color = ['#1f77b4', '#ff7f0e']
ax1.scatter(scatter_x, scatter_y, c=color)
scatter_x = [2, 1, 2]
scatter_y = [0.548, 0.544, 0.544]
color = ['#1f77b4', '#ff7f0e', '#ff7f0e']
ax2.scatter(scatter_x, scatter_y, c=color)


plt.savefig('Regularization_FB.pdf')