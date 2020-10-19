python test.py --task test \
        --batch_size 1 \
        --check_point "/home/dell/xp_workspace/sign-lang/dynamic_attn/log/reimp-conv/ep45_24.2000.pkl" \
        --eval_set test \
        --data_worker 8 \
        --video_path "/home/dell/xp_workspace/data/fullFrame-210x260px" \
        --gpu 0

#python -m ablation_analysis.similarity --task "similarity" \
#        --batch_size 1 \
#        --check_point "/home/dell/xp_workspace/sign-lang/dynamic_attn/log/reimp-conv/pretrain/ep51_23.8000.pkl" \
#        --eval_set test \
#        --data_worker 8 \
#        --video_path "/home/dell/xp_workspace/data/fullFrame-210x260px" \
#        --gpu 0

#python -m ablation_analysis.confidence --task "confidence" \
#        --batch_size 1 \
#        --check_point "/home/dell/xp_workspace/sign-lang/dynamic_attn/log/reimp-conv/pretrain/full_conv_v9_224_ep51_23.8000.pkl" \
#        --eval_set test \
#        --data_worker 8 \
#        --video_path "/home/dell/xp_workspace/data/fullFrame-210x260px" \
#        --gpu 0 \
#        --weight_decay 1e-4 \

#python test.py \
#	--task test \
#	--log_dir ./log/reimp-conv \
#	--data_worker 8 \
#	--video_path "/home/dell/xp_workspace/data/fullFrame-210x260px" \
#	--gpu 0 \
#	--check_point "/home/dell/xp_workspace/sign-lang/dynamic_attn/log/reimp-conv/ep53_26.0000.pkl" \
#	--batch_size 2 \
#	--beam_width 5