import torch
from torch.utils.data import Dataset

PI = 3.14

class TrainingData(Dataset):
    def __init__(self):
        super(TrainingData, self).__init__()
        self.x = torch.cat(
            (torch.arange(-2 * PI, 2 * PI, 1), torch.tensor([2 * PI])),
            dim=0
        ).unsqueeze(1)
        self.y = torch.sin(self.x) + torch.normal(0, 0.05, self.x.size())

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)
    
    def get_all_data(self):
        return self.x, self.y


class TestData(Dataset):
    def __init__(self):
        super(TestData, self).__init__()
        self.x = torch.arange(-2 * PI, 2 * PI, 0.005).unsqueeze(1)
        self.y = torch.sin(self.x) + torch.normal(0, 0.05, self.x.size())

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

    def get_all_data(self):
        return self.x, self.y
