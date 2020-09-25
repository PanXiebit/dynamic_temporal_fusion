#python test.py --task test \
#        --batch_size 1 \
#        --check_point "/home/panxie/workspace/sign-lang/unlikeli_ctc/log/reimp-conv/ep26_27.2000.pkl" \
#        --eval_set test \
#        --data_worker 8 \
#        --video_path "/home/panxie/workspace/sign-lang/fullFrame-210x260px" \
#        --gpu 0

python test.py \
	--task test \
	--log_dir ./log/reimp-conv \
	--data_worker 8 \
	--video_path "/home/dell/xp_workspace/data/fullFrame-210x260px" \
	--gpu 0 \
	--check_point "/home/dell/xp_workspace/sign-lang/dynamic_attn/log/reimp-conv/ep36_25.7000.pkl" \
	--batch_size 2 \
	--beam_width 5