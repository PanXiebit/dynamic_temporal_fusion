python train.py \
	--task train_v5_test \
	--log_dir ./log/reimp-conv \
	--reset_lr True \
	--learning_rate 3e-5 \
	--weight_decay 1e-4 \
	--data_worker 8 \
	--print_step 100 \
	--video_path "/home/panxie/workspace/sign-lang/fullFrame-210x260px" \
	--gpu 0 \
	--check_point "/home/panxie/workspace/sign-lang/dynamic_attn/log/reimp-conv/ep59_26.3000.pkl" \
	--batch_size 2 \
	--stage_epoch 100 \
	--beam_width 5 \
	--pretrain "/home/panxie/workspace/sign-lang/dynamic_attn/log/reimp-conv/pretrain/ep.pkl" \
  --max_epoch 100 \
#  --only_load_backbone True \
#  --freeze_cnn True \
