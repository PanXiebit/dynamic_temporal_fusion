import torch
import numpy as np
from scipy.special import binom
from scipy.misc import factorial

k = 16
prob = np.random.randn(5, 16)

def softmax(params):
    print(np.max(params, axis=1).shape)
    # if not probs, params is logits without softmax.
    params = params - np.max(params, axis=1, keepdims=True)
    params = np.exp(params)
    params = params / np.sum(params, axis=1, keepdims=True)
    return params
prob = softmax(prob)

# c = np.asarray([[(i) for i in range(0, k)]], dtype="float32")
# print(c)
# binom_coef = binom(k-1, c).astype("float32")
# print(binom_coef)

# eps = 1e-6
# def trans(px):
#     return (np.log(binom_coef) + (c * np.log(px + eps)) + ((k - 1 - c) * np.log(1. - px + eps)))
#
# print(np.exp(trans(prob)))

c = np.asarray([[(i+1) for i in range(0, k)]], dtype="float32")
cf = factorial(c)

def trans(x, tau=8):
    return ((c*np.log(x)) - x - np.log(cf)) / tau

print(np.exp(trans(prob)))