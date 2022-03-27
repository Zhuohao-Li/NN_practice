# import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from d2l import torch as d2l
import matplotlib.pyplot as plt


def f(x):
    return 3 * x**2 - 4 * x


def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h


h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical_limit={numerical_lim(f,1,h):.5f}')
    h *= 0.1


def use_svg_display():
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, tlabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X,
         Y=None,
         xlabel=None,
         ylabel=None,
         legend=None,
         xlim=None,
         ylim=None,
         xscale='linear',
         yscale='linear',
         fmts=('-', 'm--', 'g--', 'r:'),
         figsize=(3.5, 2.5),
         axes=None):
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # 如果X有⼀个轴，输出True

    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1
                or isinstance(X, list) and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


x = np.arange(-3, 3, 0.1)
# plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
# # plt.show()

plt.plot(x, f(x), 'r', x, x * 2 - 3, ':')
label = ['f(x)', 'Tangent line(x=1)']
plt.legend(label, loc='upper left')
plt.savefig(
    '/Users/edith_lzh/Desktop/Python/NN_practice/images/calculus_1.jpg')
plt.show()

x1 = np.arange(0.1, 5, 0.1)
plt.plot(x1, x1**3 - 1 / x1, 'r', x1, 4 * (x1 - 1), ':')
label = ['x1**3-1/x1', 'Tangent line(x=1)']
plt.savefig('./NN_practice/images/calculus_2.jpg')
plt.show()