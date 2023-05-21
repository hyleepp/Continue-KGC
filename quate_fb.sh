export CUDA_VISIBLE_DEVICES=0
python main.py \
    --dataset FB15K \
    --model QuatE \
    --regularizer DURA_QuatE \
    --setting active_learning \
    --batch_size 1000 \
    --init_ratio 0.9 \
    --patient 5 \
    --hidden_size 500 \
    --neg_size -1 \
    --max_completion_step 200 \
    --max_epochs 100 \
    --reg_weight 0.001 \
    --incremental_learning_method retrain \
    --active_num 10000 \
    --incremental_learning_epoch 10 \
    --update_freq 5 \
    # --pretrained_model_id logs/03_10/FB15K/QuatE_05_20_34
    
