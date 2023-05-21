export CUDA_VISIBLE_DEVICES=1
python main.py \
    --dataset WN18 \
    --model UniBi_2 \
    --regularizer DURA_UniBi_2 \
    --setting active_learning \
    --batch_size 1000 \
    --init_ratio 0.7 \
    --patient 5 \
    --hidden_size 500 \
    --neg_size -1 \
    --max_completion_step 50 \
    --max_epochs 100 \
    --reg_weight 0.01 \
    --incremental_learning_method finetune \
    --active_num 10000 \
    --incremental_learning_epoch 20 \
    --update_freq 5 \
    --sta_scale 60 \
    # --pretrained_model_id logs/03_03/FB15K/UniBi_2_20_01_27 \
    
