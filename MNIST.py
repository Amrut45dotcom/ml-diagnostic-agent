import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import pandas as pd
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

full_train = MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test = MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train, val = torch.utils.data.random_split(
    full_train,
    [54000, 6000]
)

train_loader = DataLoader(
    train,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    test,
    batch_size=64,
    shuffle=False
)

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc = nn.Linear(32 * 5 * 5, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

criterion = nn.CrossEntropyLoss()

lr_values = [0.01, 0.05, 0.1]
epochs = 5

results = []
accuracy_results = []

for lr in lr_values:

    print("\n" + "=" * 60)
    print(f"Training with learning rate = {lr}")
    print("=" * 60)

    model = Net()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        running_gradient_norm = 0.0

        for images, labels in train_loader:

            optimizer.zero_grad()

            output = model(images)

            loss = criterion(output, labels)

            loss.backward()

            total_norm = 0.0


            for parameter in model.parameters():
                if parameter.grad is not None:
                    param_norm = parameter.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2

            total_norm = total_norm ** 0.5

            running_gradient_norm += total_norm

            optimizer.step()

            running_train_loss += loss.item()

        train_loss = (
            running_train_loss /
            len(train_loader)
        )

        gradient_norm = (
            running_gradient_norm /
            len(train_loader)
        )


        model.eval()

        running_val_loss = 0.0

        with torch.no_grad():

            for images, labels in val_loader:
                output = model(images)
                loss = criterion(output, labels)
                running_val_loss += loss.item()

        val_loss = (
            running_val_loss /
            len(val_loader)
        )


        results.append({
            "lr": lr,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "gradient_norm": gradient_norm
        })

        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Gradient Norm: {gradient_norm:.4f}"
        )

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:
            output = model(images)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (
                (predicted == labels)
                .sum()
                .item()
            )

    accuracy = 100 * correct / total
    accuracy_results.append({
        "lr": lr,
        "test_accuracy": accuracy
    })

    print(
        f"Test Accuracy for LR {lr}: "
        f"{accuracy:.2f}%"
    )


# =========================================================
#  SAVE  RESULTS
# =========================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "training_results.csv",
    index=False
)
accuracy_df = pd.DataFrame(accuracy_results)

accuracy_df.to_csv(
    "accuracy_results.csv",
    index=False
)

print("\nFinal Test Accuracy Comparison")
print("--------------------------------")

for result in accuracy_results:

    print(
        f"LR = {result['lr']} "
        f"→ {result['test_accuracy']:.2f}%"
    )

for lr in lr_values:

    lr_data = results_df[
        results_df["lr"] == lr
    ]

    plt.plot(
        lr_data["epoch"],
        lr_data["train_loss"],
        marker="o",
        label=f"LR = {lr}"
    )


plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss for Different Learning Rates")
plt.legend()
plt.show()
for lr in lr_values:

    lr_data = results_df[
        results_df["lr"] == lr
    ]

    plt.plot(
        lr_data["epoch"],
        lr_data["val_loss"],
        marker="o",
        label=f"LR = {lr}"
    )


plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss for Different Learning Rates")
plt.legend()
plt.show()

for lr in lr_values:

    lr_data = results_df[
        results_df["lr"] == lr
    ]
    plt.plot(
        lr_data["epoch"],
        lr_data["gradient_norm"],
        marker="o",
        label=f"LR = {lr}"
    )


plt.xlabel("Epoch")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norm for Different Learning Rates")
plt.legend()
plt.show()