import torch

a = torch.ones(2, 2, 3, 8).long()
a[:, 0, 0, :] = 0
a[:, 0, 1, :] = 1
a[:, 0, 2, :] = 2
a[:, 1, 0, :] = 3
a[:, 1, 1, :] = 4
a[:, 1, 2, :] = 5
print(a)

b = a.permute(0, 3, 1, 2).contiguous()
print("-"*50)
print(b.shape)  # [2, 8, 2, 3]
print(b[0, :, 0, 0])
print(b[0, :, 0, 1])
print(b[0, :, 0, 2])
print(b[0, :, 1, 0])
print(b[0, :, 1, 1])
print(b[0, :, 1, 2])

c = a.reshape(-1, 3, 8).permute(0, 2, 1).contiguous()  # [4, 8 ,3]
print("-"*50)
print(c[0, :, 0])
print(c[0, :, 1])
print(c[0, :, 2])
print(c[1, :, 0])
print(c[1, :, 1])
print(c[1, :, 2])

# a = torch.zeros(6, 8).long() # [bs*t, channel]
# a[0, :] = 0
# a[1, :] = 1
# a[2, :] = 2
# a[3, :] = 3
# a[4, :] = 4
# a[5, :] = 6
#
# b = a.reshape(2, 3, 8)
# d = b.permute(0, 2, 1)
# print(b)
# print(d)
#
#
# c = a.reshape(2, 8, 3)
# print(c)

