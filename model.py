import torch.nn as nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=3),
            nn.ReLU(),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(576, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 576)
        x = self.linear_layers(x)
        return x