import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Super parameters ------------------------------------------------------------------------------------
batch_size = 64
learning_rate = 0.01
momentum = 0.5
EPOCH = 10

# Prepare dataset ------------------------------------------------------------------------------------
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


# Design model using class ------------------------------------------------------------------------------
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(
            28 * 28, 512
        )  # Flatten 28x28 input to a vector of 512 units
        self.fc2 = torch.nn.Linear(512, 128)  # First hidden layer
        self.fc3 = torch.nn.Linear(
            128, 10
        )  # Output layer with 10 units (one for each digit)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = self.fc1(x)  # the first linear layer
        x = F.relu(x)  # Activation
        x = self.fc2(x)  # the second linear layer
        x = F.relu(x)  # Activation
        x = self.fc3(x)  # No activation on the final output layer (for classification)
        return x


model = Net()

# Construct loss and optimizer ----------------------------------------------------------------------
criterion = (
    torch.nn.CrossEntropyLoss()
)  # CrossEntropy loss for multi-class classification
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, momentum=momentum
)  # SGD optimizer


# Train and Test CLASS -----------------------------------------------------------------------------------
def train(epoch):
    running_loss = 0.0
    running_total = 0
    running_correct = 0
    for batch_idx, data in enumerate(train_loader):
        inputs, target = data
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()

        # Accumulate loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, dim=1)
        running_total += target.shape[0]
        running_correct += (predicted == target).sum().item()

    # Print after each epoch
    avg_loss = running_loss / len(train_loader)
    avg_acc = 100 * running_correct / running_total
    print(
        "[%d / %d]: Training Loss: %.3f, Training Accuracy: %.2f %%"
        % (epoch + 1, EPOCH, avg_loss, avg_acc)
    )


# ... existing code ...


def add_noise(images, noise_level=0.5):
    """
    添加噪声函数
    参数:
        images: 输入数据 batch (torch.Tensor)
        noise_level: 噪声强度
    返回:
        添加噪声后的数据
    """
    noise = noise_level * torch.randn_like(images)  # 生成随机噪声
    noisy_images = images + noise  # 将噪声添加到数据中
    noisy_images = torch.clamp(noisy_images, 0.0, 1.0)  # 限制值范围在 [0, 1]
    return noisy_images


def test(epoch, noise_level=0.2):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # 添加噪声到测试数据
            noisy_images = add_noise(images, noise_level=noise_level)
            outputs = model(noisy_images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print(
        "[%d / %d]: Accuracy on test set with noise level %.2f after epoch %d: %.1f %%"
        % (epoch + 1, EPOCH, noise_level, epoch + 1, 100 * acc)
    )
    return 100 * acc


# Add EarlyStopping class after imports
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = None
        self.should_stop = False

    def __call__(self, acc):
        if self.best_acc is None:
            self.best_acc = acc
        elif acc <= self.best_acc + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_acc = acc
            self.counter = 0


if __name__ == "__main__":
    acc_list_test_noisy = []
    early_stopping = EarlyStopping(patience=3)

    for epoch in range(EPOCH):
        train(epoch)
        acc_test_noisy = test(epoch, noise_level=0.8)
        acc_list_test_noisy.append(acc_test_noisy)

        # Early stopping check
        early_stopping(acc_test_noisy)
        if early_stopping.should_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    plt.plot(range(1, len(acc_list_test_noisy) + 1), acc_list_test_noisy)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy On Noisy Test Set")
    plt.show()
