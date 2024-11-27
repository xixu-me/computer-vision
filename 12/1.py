import torch
import torch.nn as nn
import torch.optim as optim


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(2, 5)
        self.relu = nn.ReLU()
        self.output = nn.Linear(5, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x


model = SimpleNN()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

inputs = torch.abs(torch.randn(10, 2))
targets = torch.sqrt(inputs[:, 0] * inputs[:, 1]).view(-1, 1)

epochs = 10000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(inputs)

    loss = criterion(outputs, targets)

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    test_input = torch.tensor([[1.0, 1.0]])
    test_output = model(test_input)
    print(f"Test input: {test_input}, Predicted output: {test_output.item()}")
