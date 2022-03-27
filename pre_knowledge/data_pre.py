import os
import pandas as pd
import torch

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms, Alley, Price\n')
    f.write('NA, Pave, 125700\n')
    f.write('2, NA, 106000\n')
    f.write('4, NA, 178000\n')
    f.write('NA, NA, 140000\n')

data = pd.read_csv(data_file)
print(data)

## handle NaN
inputs, outputs = data.iloc[:, :2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# transfer to tensor
X, Y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(X, Y)