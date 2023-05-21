from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

file_path = "/path/to/log-directory/events.out.tfevents.xxxxxx"
# logs/03_21/FB15K/ComplEx_16_02_48
# logs/03_09/FB15K/ComplEx_14_34_26
file_path_1 = "logs/03_09/FB15K/ComplEx_14_34_26/events.out.tfevents.1678343666.psdz.74085.0"
file_path_2 = "logs/03_21/FB15K/ComplEx_16_02_48/events.out.tfevents.1679385768.psdz.4067.0"
transe_with = 'logs/03_23/FB15K/TransE_13_19_41/events.out.tfevents.1679548781.psdz.80994.0'
transe_without = 'logs/03_22/FB15K/TransE_08_22_48/events.out.tfevents.1679444568.psdz.48191.0'

loader = event_accumulator.EventAccumulator(file_path_1)
loader.Reload()
cr_1 = loader.scalars.Items('completion_ratio')
cr_1 = [x.value for x in cr_1[:200]]

loader = event_accumulator.EventAccumulator(file_path_2)
loader.Reload()
cr_2 = loader.scalars.Items('completion_ratio')
cr_2 = [x.value for x in cr_2[:200]]

loader = event_accumulator.EventAccumulator(transe_with)
loader.Reload()
transe_w = loader.scalars.Items('completion_ratio')
transe_w = [x.value for x in transe_w[:200]]

loader = event_accumulator.EventAccumulator(transe_without)
loader.Reload()
transe_wo = loader.scalars.Items('completion_ratio')
transe_wo = [x.value for x in transe_wo[:200]]

plt.plot(cr_1, label = 'ComplEx w/ SVF')
plt.plot(cr_2, label = 'ComplEx w/o SVF')
plt.plot(transe_w, label = 'TransE w/ SVF')
plt.plot(transe_wo, label = 'TransE w/o SVF')
plt.xlabel('Step')
plt.ylabel('Completion Ratio')
plt.legend()
plt.savefig('filter ablation.pdf')