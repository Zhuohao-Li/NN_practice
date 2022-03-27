from numpy import dtype, zeros_like
import torch

# basic initialization of tensor
print('task1')

x = torch.arange(12)

print(x)
print(x.shape)
print(x.numel())

X = x.reshape(3, 4)

print(X)
print(
    X.shape
)  # shape can return what's the tensor look like, how many rows and columns
print(
    X.numel()
)  # numel() can return the total elements number in a tensor, no matter how it looks like, rows and columns
# print(x.size), no this usage

z = torch.zeros((2, 3, 4))  # 3 dimensions
print(z)

o = torch.ones((3, 4, 5))
print(o)

r = torch.randn(5, 3, 4)  # generate random numbers follows guess distribution
print(r)

# combine a tensor
cc = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(cc)

## operations
print('task2')
xx = torch.arange(4)
y = torch.tensor([2, 2, 2, 2])
print(xx + y)
print(xx - y)
print(xx * y)
print(xx / y)
print(torch.exp(xx))

XX = torch.arange(16, dtype=torch.float32).reshape((4, 4))
YY = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
print(torch.cat((XX, YY), dim=0))
print(torch.cat((XX, YY), dim=1))
print(XX == YY)
print(XX.sum())
print(YY.sum())

## broadcast
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a, b)
print(a + b)

## index
print(XX[-1], XX[1:3])
XX[2, 2] = -1
print(XX)
XX[0:4, :] = -1
print(XX)

## memory control
ZZ = torch.zeros_like(YY)
print('id(ZZ):', id(ZZ))
print(XX, YY, ZZ)
ZZ[:] = XX + YY
print('id(ZZ):', id(ZZ))

## numpy
A = XX.numpy()
B = torch.tensor(A)
print(type(A))
print(type(B))