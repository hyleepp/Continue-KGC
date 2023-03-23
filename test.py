from matplotlib import pyplot as plt

max_batch = 10000
cur_batch = 1

i = 100

nums = []

for i in range(50):
    nums.append(cur_batch)
    cur_batch = min(cur_batch * 2, max_batch)

plt.yscale('log')
plt.plot(nums)
plt.xlabel('Step')
plt.ylabel('Batch Size')
plt.savefig('batch per time中文.pdf')