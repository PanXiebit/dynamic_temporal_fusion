python train.py \
	--task train \
	--log_dir ./log/reimp-conv \
	--learning_rate 1e-4 \
	--weight_decay 1e-4 \
	--data_worker 8 \
	--print_step 100 \
	--video_path "/home/dell/xp_workspace/fullFrame-210x260px" \
	--gpu 0 \
	--check_point "/home/dell/xp_workspace/sign-lang/dynamic_attn/log/reimp-conv/ep.pkl" \
	--batch_size 2 \
	--stage_epoch 100 \
	--beam_width 5 \
	#--DEBUG True
