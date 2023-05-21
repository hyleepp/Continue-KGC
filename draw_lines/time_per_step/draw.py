from matplotlib import pyplot as plt
import numpy as np

with open('draw_lines/time_per_step/data.txt', 'r') as f:
    data = f.read()
    data = data.split()
    data = data[:100]
    data = list(map(lambda x: float(x), data))

with open('draw_lines/time_per_step/data2.txt', 'r') as f:
    data2 = f.read()
    data2 = data2.split()
    data2 = data2[:100]
    data2 = list(map(lambda x: float(x), data2))

linear_model=np.polyfit(list(range(100)),data,1)
linear_model_fn=np.poly1d(linear_model)
plt.plot(data, '*', color='royalblue', ms=2, label='RF + warm-up')
plt.plot(linear_model_fn(list(range(100))), color='#6495ED', linewidth=1)
linear_model=np.polyfit(list(range(100)),data2,1)
linear_model_fn=np.poly1d(linear_model)
plt.plot(data2, '*', ms=2, color='darkorange', label='torch.topk')
plt.plot(linear_model_fn(list(range(100))), color='orange', linewidth=1)

plt.xlabel('Step')
plt.ylabel('Time(s)')
plt.legend()
plt.savefig('time per step.pdf')