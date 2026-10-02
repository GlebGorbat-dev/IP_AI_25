"""
Визуализация работы предобученной ResNet18 и кастомной СНС из ЛР1.

Запуск:
    python predict.py                 - откроется окно выбора файла изображения
    python predict.py путь/к/файлу    - классифицировать указанный файл
Если файл не выбран, берётся случайное изображение из тестовой выборки MNIST.
"""
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import datasets

from models import SimpleCNN, build_resnet18, custom_transform, resnet_transform

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RESNET_PATH = os.path.join(BASE_DIR, "mnist_resnet18.pth")
CUSTOM_PATH = os.path.join(BASE_DIR, "lab1", "mnist_cnn.pth")


def choose_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Выберите изображение цифры",
        filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"), ("Все файлы", "*.*")],
    )
    root.destroy()
    return path or None


def to_mnist_format(image):
    img = image.convert("L")
    arr = np.asarray(img, dtype=np.float32)
    if np.median(arr) > 127:
        arr = 255 - arr

    background = np.median(arr)
    arr = np.clip((arr - background) / max(arr.max() - background, 1.0), 0, 1)
    arr = np.clip((arr - 0.3) / 0.7, 0, 1) * 255
    img = Image.fromarray(arr.astype(np.uint8))

    mask = arr > 50
    if mask.any():
        rows, cols = np.where(mask)
        img = img.crop((cols.min(), rows.min(), cols.max() + 1, rows.max() + 1))

    w, h = img.size
    scale = 20.0 / max(w, h)
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(img, ((28 - img.width) // 2, (28 - img.height) // 2))
    return canvas


def load_models():
    for path in (RESNET_PATH, CUSTOM_PATH):
        if not os.path.exists(path):
            sys.exit(f"Файл модели не найден: {path}. Сначала запустите main.py.")
    resnet = build_resnet18(num_classes=10, pretrained=False)
    resnet.load_state_dict(torch.load(RESNET_PATH, map_location="cpu", weights_only=True))
    custom = SimpleCNN()
    custom.load_state_dict(torch.load(CUSTOM_PATH, map_location="cpu", weights_only=True))
    return resnet.eval(), custom.eval()


@torch.no_grad()
def classify(model, tensor):
    return torch.softmax(model(tensor.unsqueeze(0)), dim=1)[0].numpy()


def plot_probs(ax, probs, title):
    pred = int(probs.argmax())
    ax.bar(range(10), probs, color=["tab:green" if d == pred else "tab:blue" for d in range(10)])
    ax.set_xticks(range(10))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Цифра")
    ax.set_ylabel("Вероятность")
    ax.set_title(f"{title}\nРезультат: {pred} ({probs[pred]:.1%})")


def show_result(resnet, custom, original, title, save_path=None):
    prepared = to_mnist_format(original)
    resnet_probs = classify(resnet, resnet_transform(prepared))
    custom_probs = classify(custom, custom_transform(prepared))

    print(title)
    for name, probs in (("ResNet18", resnet_probs), ("Кастомная СНС (ЛР1)", custom_probs)):
        pred = int(probs.argmax())
        print(f"  {name}: цифра {pred} (уверенность {probs[pred]:.2%})")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), gridspec_kw={"width_ratios": [1, 1, 1.4, 1.4]})
    axes[0].imshow(np.asarray(original.convert("L")), cmap="gray")
    axes[0].set_title("Исходное изображение")
    axes[1].imshow(np.asarray(prepared), cmap="gray")
    axes[1].set_title("После предобработки 28x28")
    for ax in axes[:2]:
        ax.axis("off")
    plot_probs(axes[2], resnet_probs, "ResNet18 (вход 3x224x224)")
    plot_probs(axes[3], custom_probs, "Кастомная СНС из ЛР1 (вход 1x28x28)")
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def main():
    resnet, custom = load_models()
    path = sys.argv[1] if len(sys.argv) > 1 else choose_file()

    if path:
        original = Image.open(path)
        title = f"Файл: {os.path.basename(path)}"
        name = os.path.splitext(os.path.basename(path))[0]
    else:
        test_set = datasets.MNIST(DATA_DIR, train=False, download=True)
        idx = random.randrange(len(test_set))
        original, label = test_set[idx]
        title = f"Тестовое изображение MNIST №{idx}, истинная цифра: {label}"
        name = f"mnist_{idx}"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    show_result(resnet, custom, original, title, os.path.join(RESULTS_DIR, f"prediction_{name}.png"))


if __name__ == "__main__":
    main()
