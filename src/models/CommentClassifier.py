import torch.nn as nn
import torch.nn.functional as F
import torch


class CommentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_lstm_layers=1):
        super(CommentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lstms = nn.ModuleList([nn.LSTM(hidden_dim, hidden_dim, batch_first=True) for _ in range(num_lstm_layers)])
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = x.unsqueeze(1)

        for lstm in self.lstms:
            output, _ = lstm(x)
            x = output

        output = output[:, -1, :]
        output = self.fc2(output)
        return output
