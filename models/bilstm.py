import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, input_size=1024, hidden_size=256, num_layers=1):
        super(BiLSTM, self).__init__()
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size*2, 1)

    def forward(self, x):
        h, _= self.LSTM(x)
        y = torch.sigmoid(self.linear(h))
        #y = y.view(1, -1)
        return y

class BiLSTM_proj(nn.Module):
    def __init__(self, input_size=1024, hidden_size=128, num_layers=1):
        super(BiLSTM_proj, self).__init__()
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size*2, 1)

    def forward(self, x):
        h, (hidden,_)= self.LSTM(x)
        return hidden.squeeze(0).view(2,256)


def bilstm_256():
    return BiLSTM(1024, 256, 1)
