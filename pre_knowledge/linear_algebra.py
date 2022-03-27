from lib2to3.pgen2.tokenize import TokenError
from matplotlib.pyplot import axis
from numpy import dtype, float32
import torch

# vector

x = torch.arange(4)
print(x)
print(len(x))
print(x.shape)

# matrix

A = torch.arange(20).reshape(5, 4)
print('A=', A)
print('A,T=', A.T)
print(A.shape)
print(A.T.shape)

# tensor

X = torch.arange(20, dtype=torch.float32).reshape(5, 4)
Y = X.clone()
print(X)
print(X + Y)
###### Hadamard #######
print(X * Y)

# lower dimension
print(X)
print('total sum:', X.sum())
X_sum_axis0 = X.sum(axis=0)  # axis = 0 means the column of the matrix
print(X_sum_axis0)
print(X_sum_axis0.shape)
X_sum_axis1 = X.sum(axis=1)  # row
print(X_sum_axis1)
print(X_sum_axis1.shape)

print('axis=1 and axis=0:', X.sum(axis=[0, 1]))

print(X.mean() == X.sum() / X.numel())
print(X.mean(axis=1) == X.sum(axis=1) / X.shape[1])
print(X.mean(axis=1))  # each row's mean
print(X.mean(axis=0))  # each column's mean

# Non-decrease demension sum
sum_X = X.sum(axis=1, keepdims=True)
sum_Y = X.sum(axis=0, keepdims=True)
print(sum_X)
print(sum_Y)
print(X / sum_X)
print(X / sum_Y)
print(X.cumsum(axis=0))

# Dot-product
x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4, dtype=torch.float32)
print(torch.dot(x, y))

# matrix vector multiply
print(torch.mv(X, y))

# matrix matrix multiply
Y = torch.ones(4, 3)
print('Y=', Y)
print('X=', X)
print('X*Y=', torch.mm(X, Y))

# norm
###### L2 norm
u = torch.tensor([3.0, -4.0])
print('L2 norm square=', pow(torch.norm(u), 2))
###### L1 norm
print('L1 norm=', torch.abs(u).sum())
###### Fibonacci norm
print('Fibonacci norm= ', torch.norm(torch.ones((4, 9))))
