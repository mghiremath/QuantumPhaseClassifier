import torch.nn as nn
import torch.nn.functional as F
class MLPModel(nn.Module):
    def __init__(self, input_dim=370*370, hidden_dim=512, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))      # First hidden layer
        x = F.relu(self.fc2(x))      # Second hidden layer
        x = self.fc3(x)              # Output logits for 3 classes
        return x