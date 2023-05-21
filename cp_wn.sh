export CUDA_VISIBLE_DEVICES=0
python main.py \
    --dataset WN18 \
    --model CP \
    --regularizer F2 \
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
    --update_freq 5 \
    # --pretrained_model_id logs/05_11/WN18/CP_finetune
    