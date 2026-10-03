"""
Лабораторная работа №2. Конструирование моделей на базе предобученных нейронных сетей.
Вариант 6: выборка MNIST, оптимизатор Adam, предобученная архитектура ResNet18.
"""
import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from models import (IMAGENET_MEAN, IMAGENET_STD, SimpleCNN, build_resnet18,
                    custom_transform, resnet_transform)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(BASE_DIR, "mnist_resnet18.pth")
LAB1_MODEL_PATH = os.path.join(BASE_DIR, "lab1", "mnist_cnn.pth")
LAB1_HISTORY_PATH = os.path.join(BASE_DIR, "lab1", "history.json")

EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_loaders(transform, batch_size, workers):
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers,
                              pin_memory=True, persistent_workers=workers > 0)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=workers,
                             pin_memory=True, persistent_workers=workers > 0)
    return train_set, test_set, train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
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

    fig.suptitle("Обучение ResNet18 на MNIST (оптимизатор Adam)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def plot_comparison(history, lab1_history, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for hist, name, marker in [(lab1_history, "Кастомная СНС (ЛР1)", "o-"), (history, "ResNet18", "s-")]:
        epochs = range(1, len(hist["test_loss"]) + 1)
        ax1.plot(epochs, hist["test_loss"], marker, label=name)
        ax2.plot(epochs, hist["test_acc"], marker, label=name)
    ax1.set_title("Ошибка на тестовой выборке")
    ax1.set_xlabel("Эпоха")
    ax1.set_ylabel("Loss")
    ax2.set_title("Точность на тестовой выборке, %")
    ax2.set_xlabel("Эпоха")
    ax2.set_ylabel("Accuracy, %")
    for ax in (ax1, ax2):
        ax.set_xticks(range(1, max(len(history["test_loss"]), len(lab1_history["test_loss"])) + 1))
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Сравнение ResNet18 и кастомной СНС из ЛР1")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


@torch.no_grad()
def plot_test_predictions(model, test_set, device, path, count=12):
    model.eval()
    indices = random.sample(range(len(test_set)), count)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    fig, axes = plt.subplots(2, count // 2, figsize=(14, 6))
    for ax, idx in zip(axes.flat, indices):
        image, label = test_set[idx]
        probs = torch.softmax(model(image.unsqueeze(0).to(device)).float(), dim=1)[0]
        pred = probs.argmax().item()
        ax.imshow((image * std + mean).clamp(0, 1).permute(1, 2, 0))
        color = "green" if pred == label else "red"
        ax.set_title(f"True: {label}\nPred: {pred} ({probs[pred]:.1%})", color=color, fontsize=10)
        ax.axis("off")
    fig.suptitle("Классификация изображений тестовой выборки сетью ResNet18 (вход 3x224x224)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def evaluate_lab1_model(criterion, device):
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(LAB1_MODEL_PATH, map_location=device, weights_only=True))
    _, _, _, test_loader = get_loaders(custom_transform, BATCH_SIZE, 0)
    return evaluate(model, test_loader, criterion, device)


def main():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")

    train_set, test_set, train_loader, test_loader = get_loaders(resnet_transform, BATCH_SIZE, NUM_WORKERS)
    print(f"Обучающая выборка: {len(train_set)} изображений, тестовая: {len(test_set)}")
    print(f"Размер входа сети: {tuple(train_set[0][0].shape)}")

    model = build_resnet18(num_classes=10, pretrained=True).to(device)
    print(f"Новый выходной слой: {model.fc}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Число обучаемых параметров: {n_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "epoch_time": []}
    log_lines = []
    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - start
        for key, value in zip(history, (train_loss, train_acc, test_loss, test_acc, elapsed)):
            history[key].append(value)
        line = (f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
                f"| Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Time: {elapsed:.0f} s")
        print(line, flush=True)
        log_lines.append(line)

    best_epoch = int(np.argmax(history["test_acc"])) + 1
    summary = (f"ResNet18: итоговая точность на тестовой выборке {history['test_acc'][-1]:.2f}% "
               f"(ошибка классификации {100 - history['test_acc'][-1]:.2f}%). "
               f"Лучшая: {max(history['test_acc']):.2f}% на эпохе {best_epoch}")
    print(summary)
    log_lines.append(summary)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Модель сохранена: {MODEL_PATH}")

    lab1_loss, lab1_acc = evaluate_lab1_model(criterion, device)
    comparison = (f"Кастомная СНС (ЛР1): точность на тестовой выборке {lab1_acc:.2f}%, "
                  f"ошибка {lab1_loss:.4f}")
    print(comparison)
    log_lines.append(comparison)

    with open(os.path.join(RESULTS_DIR, "training_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    with open(os.path.join(RESULTS_DIR, "history.json"), "w", encoding="utf-8") as f:
        json.dump({"history": history, "params": n_params, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                   "lr": LEARNING_RATE, "device": str(device), "lab1_test_acc": lab1_acc,
                   "lab1_test_loss": lab1_loss}, f, indent=2)

    plot_history(history, os.path.join(RESULTS_DIR, "loss_accuracy.png"))
    with open(LAB1_HISTORY_PATH, encoding="utf-8") as f:
        plot_comparison(history, json.load(f)["history"], os.path.join(RESULTS_DIR, "comparison.png"))
    plot_test_predictions(model, test_set, device, os.path.join(RESULTS_DIR, "test_predictions.png"))
    plt.show()


if __name__ == "__main__":
    main()
