export CUDA_VISIBLE_DEVICES=5
python main.py \
    --dataset FB15K \
    --model TransE \
    --regularizer F2 \
    --setting active_learning \
    --batch_size 1000 \
    --init_ratio 0.1 \
    --patient 5 \
    --hidden_size 500 \
    --neg_size -1 \
    --max_epochs 100 \
    --reg_weight 0.0005\
    --incremental_learning_method retrain \
    --active_num 10000 \
    --incremental_learning_epoch 20 \
    --update_freq -1 \
    --max_batch_for_inference 15000 \
    --pretrained_model_id  logs/03_13/FB15K/TransE_20_03_41\
