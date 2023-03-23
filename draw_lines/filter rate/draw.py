from matplotlib import pyplot as plt
import seaborn
pure_cover_rate = "0.89866864	0.903020044	0.905332809	0.906812336	0.907932997	0.908514294	0.908593798	0.909863055	0.909896827"
cover_rate = "0.986452816	0.990738083	0.993079192	0.99445861	0.99534118	0.995924165	0.996355459	0.996817007	0.996757907"
pure_cover_rate = [float(x) for x in pure_cover_rate.split()]
cover_rate = [float(x) for x in cover_rate.split()]
x = list(range(1, 10))
x = [0.1 * n for n in x]

plt.subplot(2, 1, 1)
plt.plot(x, pure_cover_rate, '*-')
plt.ylabel('Cover rate (c)')

plt.subplot(2, 1, 2)
plt.plot(x, cover_rate, '*-')
plt.ylabel('Cover rate (a)')
plt.xlabel(r'Initial Ratio $\rho$')



# plt.plot(x, pure_cover_rate)
# plt.plot(x, cover_rate)
# seaborn.distplot([x, pure_cover_rate])
# seaborn.distplot([x, cover_rate])


plt.savefig('filer cover rate.pdf')



