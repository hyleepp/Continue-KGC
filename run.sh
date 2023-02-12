export CUDA_VISIBLE_DEVICES=2
python main.py \
    --dataset WN18 \
    --model RotE \
    --setting active_learning \
    --init_ratio 0.7 \
    --patient 3 \
    --hidden_size 500 \
    --neg_size -1 \
    --max_epochs 10 \
    --reg_weight 0.001 \
    --incremental_learning_method retrain \
    --active_num 5000 \
    --incremental_learning_epoch 10 \
    
