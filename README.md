
# dynamic_temporal_fusion

### baseline
与原论文一致。

|baseline||
|---|---|
|log file|/home/panxie/workspace/sign-lang/baseline/log/reimp-conv/train_seed8_log.txt, train-2_seed8_log.txt|
|best wer|epoch26,wer:29.2|
|lr schedule|halve at 40/60 epoch|
|seed|8|


|model|baseline|
|---|---|
|log file|/home/panxie/workspace/sign-lang/unlikeli_ctc/log/reimp-conv/train_origin_log.txt, train_origin_2_log.txt|
|best wer|epoch: 26, wer:27.2|
|lr schedule|halve at 40/60 epoch|


|baseline||
|---|---|
|log file|train_video_len-2_seed8_log.txt|
|best wer|epoch26,wer:27.5|
|lr schedule|halve at 40/60 epoch|
|seed|8|
len_video的计算有问题。。。改进之后总算能得到不错的效果了。

### dynamic + local self-attention
|model|full_conv_v3|
|---|---|
|module| full_conv + Encoder(1 layer)|
|log file|train_v3_pre0.00001_attn0.0001_seed8_log.txt|
|best wer|epoch: 6, wer: 27.7|
|local attn layer|1 layer|
|pretrain| load backbone and other module. and their lr is 1e-5. lr of the attention layer is 1e-4.|

可以试试 from scratch 的情况

### dynamic framing + rnn
**full_model_v5: full_conv + dynamic framing + rnn**

|model|full_model_v5, with residual connection|
|---|---|
|log file|train_v5_load_backbone_2_seed8_log.txt, train_v5_load_backbone_2-2_seed8_log.txt|
|best wer|epoch16, 29.4|
|pretrain|**load backbone and freeze.**|


|model|full_model_v5, with residual connection|
|---|---|
|module|full_conv + dynamic framing + rnn|
|log file|train_v5_scratch_seed8_log.txt|
|best wer|epoch: 69, 28.0|
|pretrain| **from scratch** |

|model|full_model_v5, **without residual connection**|
|---|---|
|log file|train_v5_scratch_no_residual_seed8_log.txt|
|best wer|epoch: 25, 36.2. Abanbon!|
|pretrain| **from scratch** |

|model|full_model_v5, with residual connection, **len_video**|
|---|---|
|module|full_conv + dynamic framing + rnn|
|log file|train_v5_video_len_seed8_log.txt|
|best wer|epoch: 59, 26.3|
|pretrain| **from scratch** |

### dynamic framing + rnn + random sample
|model| full_conv_v6, random sample after framing|
|---|---|
|module|TemporalAttention4 + conv1d + conv1d(no pooling)|
|log file|train_v6_dev_uniform_seed8_log.txt|
|best wer|epoch 86, wer: 29.9|

|model| full_conv_v6 + residual, random sample after framing|
|---|---|
|module|TemporalAttention4 + conv1d + conv1d(no pooling)|
|log file|train_v6_dev_uniform_residual_seed8_log.txt|
|best wer|epoch 17, wer: 38.7, Abanbon! overfitting.|  


### dynamic framing + rnn + random sample + tcn
|model| full_conv_v7, full_conv_v6 + tcn|
|---|---|
|module|TemporalAttention4 + tcn + tcn(no pooling)|
|log file|.txt|
|best wer|epoch , wer: |

### dynamic framing + rnn + hash_map + tcn
|model| full_conv_v8, full_conv_v7 + hash_map|
|---|---|
|module|TemporalAttention4 + tcn + tcn(no pooling)|
|log file|train_v8_seed8_log.txt|
|best wer|epoch 16, wer: 44.2. Abanbon! |

