import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleCNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(BASE_DIR, "mnist_cnn.pth")

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
SEED = 42

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=0)
    return train_set, test_set, train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, 100.0 * correct / total


def plot_history(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss")
    ax1.plot(epochs, history["test_loss"], "s-", label="Test Loss")
    ax1.set_title("Ошибка (CrossEntropyLoss)")
    ax1.set_xlabel("Эпоха")
    ax1.set_ylabel("Loss")
    ax1.set_xticks(list(epochs))
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], "o-", label="Train Accuracy")
    ax2.plot(epochs, history["test_acc"], "s-", label="Test Accuracy")
    ax2.set_title("Точность, %")
    ax2.set_xlabel("Эпоха")
    ax2.set_ylabel("Accuracy, %")
    ax2.set_xticks(list(epochs))
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.suptitle("Обучение СНС на MNIST (оптимизатор Adam)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


@torch.no_grad()
def plot_test_predictions(model, test_set, device, path, count=12):
    model.eval()
    indices = random.sample(range(len(test_set)), count)
    fig, axes = plt.subplots(2, count // 2, figsize=(14, 6))
    for ax, idx in zip(axes.flat, indices):
        image, label = test_set[idx]
        probs = torch.softmax(model(image.unsqueeze(0).to(device)), dim=1)[0]
        pred = probs.argmax().item()
        ax.imshow(image.squeeze() * MNIST_STD + MNIST_MEAN, cmap="gray")
        color = "green" if pred == label else "red"
        ax.set_title(f"True: {label}\nPred: {pred} ({probs[pred]:.1%})", color=color, fontsize=10)
        ax.axis("off")
    fig.suptitle("Примеры классификации изображений тестовой выборки")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def main():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    train_set, test_set, train_loader, test_loader = get_loaders()
    print(f"Обучающая выборка: {len(train_set)} изображений, тестовая: {len(test_set)}")

    model = SimpleCNN().to(device)
    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Число обучаемых параметров: {n_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    log_lines = []
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        line = (f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
                f"| Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        print(line)
        log_lines.append(line)

    best_epoch = int(np.argmax(history["test_acc"])) + 1
    summary = (f"Итоговая точность на тестовой выборке: {history['test_acc'][-1]:.2f}% "
               f"(ошибка классификации {100 - history['test_acc'][-1]:.2f}%). "
               f"Лучшая: {max(history['test_acc']):.2f}% на эпохе {best_epoch}")
    print(summary)
    log_lines.append(summary)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Модель сохранена: {MODEL_PATH}")

    with open(os.path.join(RESULTS_DIR, "training_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    with open(os.path.join(RESULTS_DIR, "history.json"), "w", encoding="utf-8") as f:
        json.dump({"history": history, "params": n_params, "epochs": EPOCHS,
                   "batch_size": BATCH_SIZE, "lr": LEARNING_RATE, "device": str(device)}, f, indent=2)

    plot_history(history, os.path.join(RESULTS_DIR, "loss_accuracy.png"))
    plot_test_predictions(model, test_set, device, os.path.join(RESULTS_DIR, "test_predictions.png"))
    plt.show()


if __name__ == "__main__":
    main()
