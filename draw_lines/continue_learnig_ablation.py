from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

file_path = "/path/to/log-directory/events.out.tfevents.xxxxxx"
# logs/03_21/FB15K/ComplEx_16_02_48
# logs/03_09/FB15K/ComplEx_14_34_26

retrain_path = 'logs/04_13/FB15K/ComplEx_09_06_53/events.out.tfevents.1681348013.psdz.14014.0'
# reset_path = 'logs/04_14/FB15K/ComplEx_09_35_06/events.out.tfevents.1681436106.psdz.96818.0'
finetune_path = 'logs/04_16/FB15K/ComplEx_22_02_50/events.out.tfevents.1681653770.psdz.77847.0'
naive_path = 'logs/04_12/FB15K/ComplEx_19_26_25/events.out.tfevents.1681298785.psdz.54126.0'

loader = event_accumulator.EventAccumulator(retrain_path)
loader.Reload()
retrain_data = loader.scalars.Items('completion_ratio')
retrain_data = [x.value for x in retrain_data[:200]]

# loader = event_accumulator.EventAccumulator(reset_path)
# loader.Reload()
# reset_data= loader.scalars.Items('completion_ratio')
# reset_data = [x.value for x in reset_data[:200]]

loader = event_accumulator.EventAccumulator(finetune_path)
loader.Reload()
finetune_data = loader.scalars.Items('completion_ratio')
finetune_data = [x.value for x in finetune_data[:200]]

loader = event_accumulator.EventAccumulator(naive_path)
loader.Reload()
naive_data = loader.scalars.Items('completion_ratio')
naive_data = [x.value for x in naive_data[:200]]

plt.plot(retrain_data, label = 'Retrain')
plt.plot(finetune_data, label = 'Finetune')
# plt.plot(reset_data, label = 'Reset')
plt.plot(naive_data, label = 'Naive')
plt.xlabel('Step')
plt.ylabel('Completion Ratio')
plt.legend()
plt.show()
plt.savefig('continue learning comparison.pdf')