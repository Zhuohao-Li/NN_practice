import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
## generate a data set


def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器。"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)
## generate a data iteration by load_array()

print(next(iter(data_iter)))
## using next to print the first row of data_iter
## output includes a feature(2), a label

## define model
net = nn.Sequential(nn.Linear(2, 1))
#### 2 means input size, 1 means output size

# initialize
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

###### print(net)

## loss function
loss = nn.MSELoss()

## algorithm
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# train
num_epochs = 30
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()  # clear gradients
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')