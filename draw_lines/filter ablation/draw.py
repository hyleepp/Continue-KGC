from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

file_path = "/path/to/log-directory/events.out.tfevents.xxxxxx"
# logs/03_21/FB15K/ComplEx_16_02_48
# logs/03_09/FB15K/ComplEx_14_34_26
file_path_1 = "logs/03_09/FB15K/ComplEx_14_34_26/events.out.tfevents.1678343666.psdz.74085.0"
file_path_2 = "logs/03_21/FB15K/ComplEx_16_02_48/events.out.tfevents.1679385768.psdz.4067.0"
loader = event_accumulator.EventAccumulator(file_path_1)
loader.Reload()

cr_1 = loader.scalars.Items('completion_ratio')
cr_1 = [x.value for x in cr_1[:200]]

loader = event_accumulator.EventAccumulator(file_path_2)
loader.Reload()
cr_2 = loader.scalars.Items('completion_ratio')
cr_2 = [x.value for x in cr_2[:200]]
plt.plot(cr_1)
plt.plot(cr_2)
plt.savefig('filter ablation.pdf')