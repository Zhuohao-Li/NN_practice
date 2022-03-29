from cProfile import label
from cmath import sqrt
import math
import time
from matplotlib.pyplot import legend, ylabel
import numpy as np
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

n = 10000
a = torch.ones(n)
b = torch.ones(n)

c = torch.zeros(n)


def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)


x = np.arange(-7, 7, 0.01)
# params = [(0, 1), (0, 2), (3, 1)]
# d2l.plt.plot(x, [normal(x, mu, sigma) for mu, sigma in params],
#              xlabel='x',
#              ylabel='p(x)',
#              figsize=(4.5, 2.5),
#              legend=[f'mean{mu}, std{sigma}' for mu, sigma in params])
# d2l.plt.savefig('./NN_practice/images/linear1.jpg')
# d2l.plt.show()
params = [(0, 1), (0, 2), (3, 1)]
d2l.plt.plot(x, normal(x, 0, 1), 'r', x, normal(x, 0, 2), 'b', x,
             normal(x, 3, 1), 'g')
label = ['mean:0, sigma=1', 'mean=0, sigma=2', 'mean=3, sigma=1']
d2l.plt.legend(label, loc='upper left')
d2l.plt.gca().set_xlabel('x')
d2l.plt.gca().set_ylabel('p(x)')
d2l.plt.savefig('./NN_practice/images/linear1.jpg')
d2l.plt.show()
