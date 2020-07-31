import torch
import torch.nn as nn
import torch.nn.functional as F

class PolynomialModel(nn.Module):
    def __init__(self, order):
        super(PolynomialModel, self).__init__()
        self.order = order
        self.layer = nn.Linear(order, 1)
        self.layer.weight.data.uniform_(-1e-3, 1e-3)
        self.mean, self.sd = self.get_normalization_factors()

    def get_normalization_factors(self):
        d = torch.range(-2 * 3.14, 2 * 3.14, 0.005).unsqueeze(1)
        dd = torch.cat(
            [d**(p+1) for p in range(self.order)],
            dim=1
        ) / (2 * 3.14) # force x to lie between -1 and 1
        return(torch.mean(dd, dim=0), torch.std(dd, dim=0))

    def forward(self, x):
        input_data = torch.cat(
            [x**(p+1) for p in range(self.order)],
            dim=1
        ) / (2 * 3.14)
        input_data = (input_data - self.mean) / self.sd
        return self.layer(input_data)

