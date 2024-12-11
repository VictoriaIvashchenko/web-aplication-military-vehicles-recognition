from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib import use as matplotlib_use
import os

# Встановлення бекенду Matplotlib для роботи без графічного інтерфейсу
matplotlib_use('Agg')


def process_image_with_yolo(input_path, output_path, model_path='best60.pt'):
    """
    Виконує обробку зображення за допомогою моделі YOLO та зберігає результат.

    Parameters:
        input_path (str): Шлях до вхідного зображення.
        output_path (str): Шлях для збереження обробленого зображення.
        model_path (str): Шлях до моделі YOLO, за замовчуванням 'best60.pt'.

    Returns:
        None
    """
    # Завантаження моделі YOLO
    model = YOLO(model_path)

    # Виконання передбачення
    results = model(input_path)

    # Отримання результатів
    result = results[0]

    # Нанесення рамок на зображення
    img_with_boxes = result.plot()

    # Збереження обробленого зображення
    plt.imshow(img_with_boxes)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
