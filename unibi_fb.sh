export CUDA_VISIBLE_DEVICES=4
python main.py \
    --dataset FB15K \
    --model UniBi_2 \
    --regularizer DURA_UniBi_2 \
    --setting active_learning \
    --batch_size 1000 \
    --init_ratio 0.9 \
    --patient 5 \
    --hidden_size 500 \
    --neg_size -1 \
    --max_epochs 100 \
    --reg_weight 2 \
    --incremental_learning_method retrain \
    --active_num 10000 \
    --incremental_learning_epoch 10 \
    --update_freq 10 \
    --sta_scale 60 \
    --pretrained_model_id logs/03_03/FB15K/UniBi_2_20_01_27 \
    
