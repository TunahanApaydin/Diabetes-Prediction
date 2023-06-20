import torch.nn as nn

class DiabetesPredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(15, 10)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(10, 8)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(8, 6)
        self.act3 = nn.ReLU()
        self.hidden4 = nn.Linear(6, 4)
        self.act4 = nn.ReLU()
        self.output = nn.Linear(4, 2)
 
    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.act4(self.hidden4(x))
        x = self.output(x)
        return x