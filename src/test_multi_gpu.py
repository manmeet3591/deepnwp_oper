import torch
import torch.nn as nn

# Define a simple neural network model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = self.fc(x)
        return x

# Check the number of GPUs available
num_gpus = torch.cuda.device_count()
print("Number of GPUs available:", num_gpus)

# Create an instance of the model
model = SimpleNet()

# If more than one GPU is available, use DataParallel
if num_gpus > 1:
    model = nn.DataParallel(model)

# Move the model to GPU if available
if torch.cuda.is_available():
    model.cuda()

# Example input (random tensor of size (batch_size, channels, height, width))
input = torch.randn(64, 1, 28, 28)  # Example for a batch of 64 images of size 28x28 with 1 channel

# Move the input to GPU if available
if torch.cuda.is_available():
    input = input.cuda()

# Forward pass
output = model(input)
print("Output size:", output.size())
