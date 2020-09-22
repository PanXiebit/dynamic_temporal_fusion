# import torch
#
# a = torch.ones(2, 2, 3, 8).long()
# a[:, 0, 0, :] = 0
# a[:, 0, 1, :] = 1
# a[:, 0, 2, :] = 2
# a[:, 1, 0, :] = 3
# a[:, 1, 1, :] = 4
# a[:, 1, 2, :] = 5
# print(a)
#
# b = a.permute(0, 3, 1, 2).contiguous()
# print("-"*50)
# print(b.shape)  # [2, 8, 2, 3]
# print(b[0, :, 0, 0])
# print(b[0, :, 0, 1])
# print(b[0, :, 0, 2])
# print(b[0, :, 1, 0])
# print(b[0, :, 1, 1])
# print(b[0, :, 1, 2])
#
# c = a.reshape(-1, 3, 8).permute(0, 2, 1).contiguous()  # [4, 8 ,3]
# print("-"*50)
# print(c[0, :, 0])
# print(c[0, :, 1])
# print(c[0, :, 2])
# print(c[1, :, 0])
# print(c[1, :, 1])
# print(c[1, :, 2])

import torch
import torch.nn.functional as F

src = torch.arange(25, dtype=torch.float).reshape(1, 1, 5, 5).requires_grad_()  # 1 x 1 x 5 x 5 with 0 ... 25
indices = torch.tensor([[0, 0], [2, 2]], dtype=torch.float).reshape(1, 1, -1, 2)  # 1 x 1 x 2 x 2
output = F.grid_sample(src, indices)
print(src)
print(indices)
print(output)  # tensor([[[[  0.,  12.]]]])

