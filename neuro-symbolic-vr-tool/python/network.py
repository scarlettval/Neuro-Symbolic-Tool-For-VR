import torch
import torch.nn as nn

# Force CPU execution
device = torch.device("cpu")

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))

if __name__ == "__main__":
    model = SimpleNN().to(device)
    sample_input = torch.tensor([[1.0, 2.0]]).to(device)
    output = model(sample_input)
    print("Neural Network Output:", output.item())
