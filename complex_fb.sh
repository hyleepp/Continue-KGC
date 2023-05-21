export CUDA_VISIBLE_DEVICES=5
python main.py \
    --dataset FB15K \
    --model ComplEx \
    --regularizer DURA_W \
    --setting active_learning \
    --batch_size 1000 \
    --init_ratio 0.9 \
    --patient 5 \
    --hidden_size 500 \
    --neg_size -1 \
    --max_epochs 100 \
    --reg_weight 0.0005\
    --incremental_learning_method finetune \
    --active_num 10000 \
    --incremental_learning_epoch 20 \
    --update_freq 10 \
    --pretrained_model_id logs/03_09/FB15K/ComplEx_14_34_26
    # --pretrained_model_id logs/02_16/FB15K/RESCAL_16_59_19 \
    
