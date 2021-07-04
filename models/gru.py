import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_size=1024, hidden_size=256, num_layers=1):
        super(GRU, self).__init__()
        self.LSTM = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                            bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size*2, 1)

    def forward(self, x):
        h, _= self.LSTM(x)
        y = torch.sigmoid(self.linear(h))
        #y = y.view(1, -1)
        return y

class GRU_proj(nn.Module):
    def __init__(self, input_size=1024, hidden_size=256, num_layers=1):
        super(GRU_proj, self).__init__()
        self.LSTM = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                            bidirectional=True, batch_first=True)

    def forward(self, x):
        h, hidden = self.LSTM(x)
        m = hidden.sum(0)
        return m


def bigru_256():
    return GRU(1024, 256, 1)