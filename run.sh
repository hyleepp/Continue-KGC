export CUDA_VISIBLE_DEVICES=0
if [ ! -n "$1" ]; then
    # default
    dataset=WN18
elif [ $1  == "fb" ]; then
    dataset="FB15K"
elif [ $1 == "wn" ]; then
    dataset="WN18"
elif [ $1 == "wiki" ]; then
    dataset="wikikg-v2"
fi
python main.py \
    --dataset $dataset \
    --model RotE \
    --regularizer F2 \
    --setting active_learning \
    --batch_size 1000 \
    --init_ratio 0.9 \
    --patient 5 \
    --hidden_size 500 \
    --neg_size -1 \
    --max_epochs 100 \
    --reg_weight 0.001\
    --incremental_learning_method retrain \
    --active_num 10000 \
    --incremental_learning_epoch 5 \
    --update_freq 5 \
    # --pretrained_model_id logs/02_16/FB15K/RESCAL_16_59_19 \
    
