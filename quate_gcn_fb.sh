export CUDA_VISIBLE_DEVICES=2
python main.py \
    --dataset FB15K \
    --model QuatE \
    --gcn_type RGCN \
    --regularizer F2 \
    --setting active_learning \
    --batch_size 1000 \
    --init_ratio 0.9 \
    --patient 5 \
    --hidden_size 400 \
    --neg_size -1 \
    --max_epochs 100 \
    --reg_weight 0.005\
    --incremental_learning_method retrain \
    --active_num 10000 \
    --incremental_learning_epoch 20 \
    --update_freq -1 \
    ## --pretrained_model_id logs/03_08/FB15K/QuatE_17_26_28
    
