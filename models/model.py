import torch.nn as nn
import torch.nn.functional as F

# A simple model, but forward must return a pair of outputs representing the model output first, then a 0 -> 1 scale output via sigmoid
class Model(nn.Module):
  def __init__(self, in_features: int):
    super().__init__()
    self.fc1 = nn.Linear(in_features=in_features, out_features=1)
    self.relu = nn.ReLU()
  
  def forward(self, x):
    logits = self.relu(self.fc1(x))
    probabilities = F.sigmoid(logits)
    
    return logits, probabilities

