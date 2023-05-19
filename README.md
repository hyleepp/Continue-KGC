# Progressive Knowledge Graph Completion

Official code repository for paper "Progressive Knowledge Graph Completion".

## Requirements

It can be viewed in requirements.txt.

All experiments can be run on a single 1080ti(11GB) or more powerful device. If there is not enough memory during the inference, you can adjust `max_batch_for_inference` parameter to make it smaller.

## How to run

For example, to run the QuatE model on FB15k, quate_fb.sh can be executed.

```bash
bash quate_fb.sh
```

In the script file, you can modify various parameters. Here are a few unique parameters

```
--init_ratio # The proportion of $\mathcal{K}_{known}$ in our paper, 0.7 on WN18 and 0.9 on FB15k.
--max_completion_step # In our experimental setup, we set it to 200 on the FB15k dataset and 50 on the WN18 dataset.
--max_epochs # Maximum number of training epochs.
--regularizer # The available options can be found in optimization/Regularizer.py.
--update_freq # The step interval for using continuous learning method, if not included, is set to -1.
--incremental_learning_epoch # Training epochs for continuous learning methods
```

If you are interested in other models and datasets, you can view and run other scripts in the form of "model_dataset.sh".

When the progressive completion is finished, the metrics can be calculated using the log file created by tensorboard. Just run the evaluation_fb.sh or evaluation_wn.sh.

```bash
bash evaluation_fb.sh
```

