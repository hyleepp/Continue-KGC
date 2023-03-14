export CUDA_VISIBLE_DEVICES=0
python main.py \
    --dataset FB15K \
    --model RESCAL \
    --regularizer DURA_RESCAL \
    --setting active_learning \
    --batch_size 1000 \
    --init_ratio 0.9 \
    --patient 5 \
    --hidden_size 100 \
    --neg_size -1 \
    --max_epochs 100 \
    --reg_weight 0.0005\
    --incremental_learning_method retrain \
    --active_num 10000 \
    --incremental_learning_epoch 20 \
    --update_freq -1 \
    # --pretrained_model_id logs/02_16/FB15K/RESCAL_16_59_19 \
    
