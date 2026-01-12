import torch.nn as nn

# A simple feedforward neural network (MLP)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                         # Flatten input from [2, 28, 28] to [2*28*28]
            nn.Linear(2 * 28 * 28, 256),          # Fully connected layer with 256 units
            nn.ReLU(),                            # ReLU activation
            nn.Linear(256, 1)                     # Output layer (binary classification)
        )

    def forward(self, x):
        return self.net(x)
