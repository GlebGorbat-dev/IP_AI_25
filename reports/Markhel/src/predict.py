import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import datasets

from model import SimpleCNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mnist_cnn.pth")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


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
        filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.gif"), ("Все файлы", "*.*")],
    )
    root.destroy()
    return path or None


def preprocess(image):

    img = image.convert("L")
    arr = np.asarray(img, dtype=np.float32)

    if np.median(arr) > 127:
        img = ImageOps.invert(img)
        arr = np.asarray(img, dtype=np.float32)

    background = np.median(arr)
    arr = np.clip((arr - background) / max(arr.max() - background, 1.0), 0, 1) * 255
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

    x = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy((x - MNIST_MEAN) / MNIST_STD).unsqueeze(0).unsqueeze(0)
    return tensor, x


def load_model():
    if not os.path.exists(MODEL_PATH):
        sys.exit("Файл модели не найден. Сначала запустите main.py для обучения сети.")
    model = SimpleCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def show_result(model, original, prepared, tensor, title, save_path=None):
    probs = torch.softmax(model(tensor), dim=1)[0].numpy()
    pred = int(probs.argmax())
    print(f"{title}\nПредсказанная цифра: {pred} (уверенность {probs[pred]:.2%})")
    for digit, p in enumerate(probs):
        print(f"  {digit}: {p:.4f}")

    feature_maps = model.features[1](model.features[0](tensor))[0]

    fig = plt.figure(figsize=(14, 5.5))
    grid = fig.add_gridspec(2, 11)

    ax = fig.add_subplot(grid[0, 0:3])
    ax.imshow(original, cmap="gray")
    ax.set_title("Исходное изображение")
    ax.axis("off")

    ax = fig.add_subplot(grid[0, 3:6])
    ax.imshow(prepared, cmap="gray")
    ax.set_title("Вход сети 28x28")
    ax.axis("off")

    ax = fig.add_subplot(grid[0, 6:11])
    colors = ["tab:green" if d == pred else "tab:blue" for d in range(10)]
    ax.bar(range(10), probs, color=colors)
    ax.set_xticks(range(10))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Цифра")
    ax.set_ylabel("Вероятность")
    ax.set_title(f"Результат: {pred} ({probs[pred]:.1%})")

    for i in range(11):
        ax = fig.add_subplot(grid[1, i])
        ax.imshow(feature_maps[i].numpy(), cmap="viridis")
        ax.axis("off")
        if i == 0:
            ax.set_title("Карты признаков Conv1", loc="left", fontsize=10)

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def main():
    model = load_model()
    path = sys.argv[1] if len(sys.argv) > 1 else choose_file()

    if path:
        original = Image.open(path)
        tensor, prepared = preprocess(original)
        title = f"Файл: {os.path.basename(path)}"
        original = np.asarray(original.convert("L"))
    else:
        test_set = datasets.MNIST(DATA_DIR, train=False, download=True)
        idx = random.randrange(len(test_set))
        original, label = test_set[idx]
        tensor, prepared = preprocess(original)
        title = f"Тестовое изображение MNIST №{idx}, истинная цифра: {label}"
        original = np.asarray(original)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(path))[0] if path else "random_test"
    show_result(model, original, prepared, tensor, title,
                os.path.join(RESULTS_DIR, f"prediction_{name}.png"))


if __name__ == "__main__":
    main()
