from flask import Flask, render_template, request
import os
from yolo_processor import process_image_with_yolo  # Імпортуємо функцію з іншого файлу

app = Flask(__name__)
UPLOAD_FOLDER = 'static/processed/'
MODEL_PATH = 'best60.pt'  # Шлях до вашої YOLO моделі
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Головна сторінка
@app.route('/')
def index():
    return render_template('index.html')

# Обробка завантаження зображення
@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    # Збереження оригінального зображення
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
    file.save(input_path)

    # Шлях для збереження обробленого зображення
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')

    # Обробка зображення за допомогою YOLO
    process_image_with_yolo(input_path, output_path, model_path=MODEL_PATH)

    return render_template('index.html', input_image='processed/input.jpg', output_image='processed/output.jpg')

if __name__ == '__main__':
    app.run(debug=True)
