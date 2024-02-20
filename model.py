import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class heavynetwork(nn.Module):
  def __init__(self, h1= 1000, h2 = 2000):
    super(heavynetwork, self).__init__()
    self.l1 = nn.Linear(28*28, h1)
    self.l2 = nn.Linear(h1, h2)
    self.l3 = nn.Linear(h2, 10)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = x.view(-1, 28*28)
    x = self.relu(self.l1(x))
    x = self.relu(self.l2(x))
    x = self.l3(x)
    return x