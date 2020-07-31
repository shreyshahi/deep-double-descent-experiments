import torch
import torch.nn as nn
import torch.nn.functional as F

class PolynomialModel(nn.Module):
    def __init__(self, order):
        super(PolynomialModel, self).__init__()
        self.order = order
        self.layer = nn.Linear(order, 1)
        self.layer.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, x):
        x = x / (2 * 3.14) # forces everything between -1 and 1
        input_data = torch.cat(
            [x**(p+1) for p in range(self.order)],
            dim=1
        )
        return self.layer(input_data)

