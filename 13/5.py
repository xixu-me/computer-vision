import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Super parameters
batch_size = 64
learning_rate = 0.01
momentum = 0.5
EPOCH = 10

# Prepare dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_dataset = datasets.MNIST(
    root="./data/mnist", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data/mnist", train=False, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class Net3Layer(torch.nn.Module):
    def __init__(self):
        super(Net3Layer, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net9Layer(torch.nn.Module):
    def __init__(self):
        super(Net9Layer, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 256)
        self.fc4 = torch.nn.Linear(256, 256)
        self.fc5 = torch.nn.Linear(256, 128)
        self.fc6 = torch.nn.Linear(128, 128)
        self.fc7 = torch.nn.Linear(128, 64)
        self.fc8 = torch.nn.Linear(64, 64)
        self.fc9 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)
        return x


# Create two models
model_3layer = Net3Layer()
model_9layer = Net9Layer()

# Create optimizers
optimizer_3layer = torch.optim.SGD(
    model_3layer.parameters(), lr=learning_rate, momentum=momentum
)
optimizer_9layer = torch.optim.SGD(
    model_9layer.parameters(), lr=learning_rate, momentum=momentum
)

criterion = torch.nn.CrossEntropyLoss()


def train(epoch, model, optimizer, name=""):
    model.train()
    running_loss = 0.0
    running_total = 0
    running_correct = 0

    for batch_idx, (inputs, target) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, dim=1)
        running_total += target.shape[0]
        running_correct += (predicted == target).sum().item()

    avg_loss = running_loss / len(train_loader)
    avg_acc = 100 * running_correct / running_total
    print(
        f"[{epoch + 1} / {EPOCH}]: {name} Training Loss: {avg_loss:.3f}, Training Accuracy: {avg_acc:.2f} %"
    )


def test(epoch, model, name=""):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(
        f"[{epoch + 1} / {EPOCH}]: {name} Accuracy on test set after epoch {epoch + 1}: {acc:.1f} %"
    )
    return acc


if __name__ == "__main__":
    acc_list_3layer = []
    acc_list_9layer = []

    for epoch in range(EPOCH):
        train(epoch, model_3layer, optimizer_3layer, "3-Layer")
        train(epoch, model_9layer, optimizer_9layer, "9-Layer")

        acc_3layer = test(epoch, model_3layer, "3-Layer")
        acc_9layer = test(epoch, model_9layer, "9-Layer")

        acc_list_3layer.append(acc_3layer)
        acc_list_9layer.append(acc_9layer)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCH + 1), acc_list_3layer, label="3-Layer Network", marker="o")
    plt.plot(range(1, EPOCH + 1), acc_list_9layer, label="9-Layer Network", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy On TestSet (%)")
    plt.title("Comparison of Network Depths")
    plt.legend()
    plt.grid(True)
    plt.show()
