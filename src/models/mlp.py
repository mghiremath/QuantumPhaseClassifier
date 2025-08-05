import torch.nn as nn
import torch.nn.functional as F
class MLPModel(nn.Module):
    def __init__(self, input_dim=370*370, hidden_dim=32, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)              # Output logits for 3 classes
        return x