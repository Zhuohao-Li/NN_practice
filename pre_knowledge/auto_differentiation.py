from importlib.metadata import requires
import torch

### eg1
x = torch.arange(4.0, requires_grad=True)
print('x=', x)
x.grad

y = 2 * torch.dot(x, x)
print('y=', y)

y.backward()
print('y=2xTx grad:', x.grad)

## eg2
x.grad.zero_()  # clear recent values
y = x.sum()
y.backward()
print(x.grad)

## eg3
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad == u)

x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)


## eg4
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print('a grad:', a.grad == d / a)
