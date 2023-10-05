import torch
import torch.nn as nn
from torch.autograd.functional import jacobian

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create a batch of input data
x = torch.randn(1, 2)

# Create an instance of the neural network
net = Net()
print(net(x))
# Calculate the Jacobian of the network outputs with respect to its parameters
params = tuple(list(net.parameters()))
jac = torch.autograd.grad(net(x), params)

# Print the shape of the Jacobian
#print(jac.shape)
print(jac)
