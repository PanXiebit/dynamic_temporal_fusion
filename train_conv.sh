CUDA_VISIBLE_DEVICES=0

python train.py \
	--task train_v10_win0_win4_small \
	--log_dir ./log/reimp-conv \
	--bn_momentum 0.1 \
	--learning_rate 1e-4 \
	--weight_decay 1e-4 \
	--data_worker 8 \
	--print_step 100 \
	--video_path "/home/dell/xp_workspace/data/fullFrame-210x260px" \
	--gpu 0 \
	--check_point "/home/dell/xp_workspace/sign-lang/dynamic_attn/log/reimp-conv/ep.pkl" \
	--batch_size 2 \
	--stage_epoch 100 \
	--beam_width 5 \
	--pretrain "/home/dell/xp_workspace/sign-lang/dynamic_attn/log/reimp-conv/pretrain/ep.pkl" \
  --max_epoch 100 \
#  --DEBUG True
  #--only_load_backbone True \
  #--freeze_cnn True \