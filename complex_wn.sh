export CUDA_VISIBLE_DEVICES=2
python main.py \
    --dataset WN18 \
    --model ComplEx \
    --regularizer DURA_W \
    --setting active_learning \
    --batch_size 1000 \
    --init_ratio 0.7 \
    --patient 5 \
    --hidden_size 500 \
    --neg_size -1 \
    --max_completion_step 50 \
    --max_epochs 100 \
    --reg_weight 0.001\
    --incremental_learning_method retrain \
    --active_num 10000 \
    --incremental_learning_epoch 20 \
    --update_freq -1 \
    # --pretrained_model_id logs/03_09/FB15K/ComplEx_14_34_26
    
