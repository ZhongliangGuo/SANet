export CONTENT_DATA="/workspace/datasets/COCO"
export STYLE_DATA="/workspace/datasets/PainterByNumber"
export LOGDIR="./logs"

script_path=$(readlink -f "$0")
mkdir -p "$LOGDIR"
cp "$script_path" "$LOGDIR"

python train.py \
--content_data_dir $CONTENT_DATA \
--style_data_dir $STYLE_DATA \
--log_dir $LOGDIR \
--lr 1e-4 \
--lr_decay 5e-5 \
--batch_size 16 \
--style_weight 3.0 \
--content_weight 1.0 \
--id1_weight 50.0 \
--id2_weight 1.0 \
--random_seed 3407
