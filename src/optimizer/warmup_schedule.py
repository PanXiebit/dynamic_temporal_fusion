
import torch
import math
import matplotlib.pyplot as plt

def warm_up_lr(step, warmup_steps=10000):
    arg1 = 1 / math.sqrt(step)
    arg2 = step * (warmup_steps ** -1.5)
    return 0.015 * min(arg1, arg2)


if __name__ == "__main__":
    lr = []
    for step in range(1, 2800*60):
        lr.append(warm_up_lr(step))
    plt.plot(lr)
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.show()