import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(signals.transpose(1, 2))
        return self.fc(output[:, -1, :])
