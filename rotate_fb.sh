export CUDA_VISIBLE_DEVICES=1
python main.py \
    --dataset FB15K \
    --model RotatE \
    --regularizer DURA_W \
    --setting active_learning \
    --batch_size 1000 \
    --init_ratio 0.9 \
    --patient 5 \
    --hidden_size 500 \
    --neg_size -1 \
    --max_completion_step 200 \
    --max_epochs 100 \
    --reg_weight 0.0\
    --incremental_learning_method retrain \
    --active_num 10000 \
    --incremental_learning_epoch 20 \
    --update_freq -1 \
    # --pretrained_model_id save_model/RotatE_DURA_W_0.0 \
    