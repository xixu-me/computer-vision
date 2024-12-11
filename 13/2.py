import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Super parameters
batch_size_small = 64
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

batch_size_full = len(train_dataset)
train_loader_small = DataLoader(
    train_dataset, batch_size=batch_size_small, shuffle=True
)
train_loader_full = DataLoader(train_dataset, batch_size=batch_size_full, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size_small, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Create two models
model_small_batch = Net()
model_full_batch = Net()

# Create optimizers
optimizer_small = torch.optim.SGD(
    model_small_batch.parameters(), lr=learning_rate, momentum=momentum
)
optimizer_full = torch.optim.SGD(
    model_full_batch.parameters(), lr=learning_rate, momentum=momentum
)

criterion = torch.nn.CrossEntropyLoss()


def train(epoch, model, optimizer, loader, name=""):
    model.train()
    running_loss = 0.0
    running_total = 0
    running_correct = 0

    for batch_idx, (inputs, target) in enumerate(loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, dim=1)
        running_total += target.shape[0]
        running_correct += (predicted == target).sum().item()

    avg_loss = running_loss / len(loader)
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
    acc_list_small = []
    acc_list_full = []

    for epoch in range(EPOCH):
        train(
            epoch, model_small_batch, optimizer_small, train_loader_small, "Small Batch"
        )
        train(epoch, model_full_batch, optimizer_full, train_loader_full, "Full Batch")

        acc_small = test(epoch, model_small_batch, "Small Batch")
        acc_full = test(epoch, model_full_batch, "Full Batch")

        acc_list_small.append(acc_small)
        acc_list_full.append(acc_full)

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, EPOCH + 1),
        acc_list_small,
        label=f"Batch Size={batch_size_small}",
        marker="o",
    )
    plt.plot(range(1, EPOCH + 1), acc_list_full, label="Full Batch", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy On TestSet (%)")
    plt.title("Comparison of Batch Sizes")
    plt.legend()
    plt.grid(True)
    plt.show()
