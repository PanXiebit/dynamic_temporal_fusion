import numpy as np
from six.moves import xrange


def get_alignment(params, seq, blank=0, is_prob=True):
    """
    params: [vocab_size, T], logits.softmax(-1). T is the frame number.
    seq: [seq_len] the length of output.
    """

    seqLen = seq.shape[0]  # Length of label sequence (# phones)
    L = 2 * seqLen + 1  # Length of label sequence with blanks, extended label: l'.
    T = params.shape[1]  # Length of utterance (time)

    # # transfer logits to probability
    if not is_prob:
        # if not probs, params is logits without softmax.
        params = params - np.max(params, axis=0)
        params = np.exp(params)
        params = params / np.sum(params, axis=0)

    # map: l' x T.
    alphas = np.zeros((L, T))  # forward probability

    # Initialize alphas and forward pass
    # 初始条件：T=0时，只能为 blank 或 seq[0]
    alphas[0, 0] = params[blank, 0]
    alphas[1, 0] = params[seq[0], 0]
    # T=0， alpha[2:, 0] = 0. the other label's probability is 0.

    for t in xrange(1, T):
        # 第一个循环： 计算每个时刻所有可能节点的概率
        start = max(0, L - 2 * (T - t))    # 对于时刻 t, 其可能的节点.与公式2一致。
        end = min(2 * t + 2, L)            # 对于时刻 t，最大节点范围不可能超过 2t+2
        for s in xrange(start, end):
            l = int((s - 1) / 2)

            # blank，节点s在偶数位置，意味着s为 blank
            if s % 2 == 0:
                if s == 0: # 初始位置，单独讨论
                    alphas[s, t] = alphas[s, t - 1] * params[blank, t]
                else:
                    alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * params[blank, t]

            # s为奇数，非空
            # l = (s-1/2) 就是奇数 s 所对应的 lable字符
            # l-1 = ((s-2)-1)/2 = (s-1)/2-1  就是 s-2 对应的lable字符
            elif s == 1 or seq[l] == seq[l - 1]:
                alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * params[seq[l], t]
            else:
                alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1] + alphas[s - 2, t - 1]) \
                               * params[seq[l], t]

        # normalize at current time (prevent underflow)
        c = np.sum(alphas[start:end, t])
        alphas[start:end, t] = alphas[start:end, t] / c

    # backtrace to get the max probability path.
    alignment = np.zeros((T,), dtype=np.int32)   # [T]

    ext_labels = []
    for i in range(seq.shape[0]):
        ext_labels.extend([blank, seq[i]])
    ext_labels.extend([blank])

    betas = np.zeros((L, T))  # backward probability

    # the last time step:
    betas[-1, -1] = alphas[-1, -1]
    betas[-2, -1] = alphas[-2, -1]
    ids = np.argmax(betas[:, -1])
    alignment[-1] = ext_labels[ids]

    for t in range(T-2, -1, -1):
        l = int((ids-1) / 2) # 如果为奇数，这是其对应的非blank的label
        if ids % 2 == 0:   # blank
            betas[ids, t] = alphas[ids, t]
            betas[ids-1, t] = alphas[ids-1, t]
            ids = np.argmax(betas[:, t])
        elif seq[l] == seq[l - 1]:
            betas[ids, t] = alphas[ids, t]
            betas[ids - 1, t] = alphas[ids - 1, t]
            ids = np.argmax(betas[:, t])
        else:
            betas[ids, t] = alphas[ids, t]
            betas[ids - 1, t] = alphas[ids - 1, t]
            betas[ids - 2, t] = alphas[ids - 2, t]
            ids = np.argmax(betas[:, t])
        alignment[t] = ext_labels[ids]
    return alignment


if __name__ == "__main__":
    prob = np.random.randn(1238, 50)
    seq = np.array([20, 8, 19, 4, 123, 5, 9, 100])
    alignment = get_alignment(prob, seq, blank=0, is_prob=False)
    print(alignment)