import pandas as pd
import numpy as np
import os
from PIL import Image

# Cargar el dataset
fer_csv = '/home/abner/Escritorio/fer/motion_detection/dataset/archive(3)/fer2013.csv'  # Cambia esta ruta al archivo descargado
data = pd.read_csv(fer_csv)

# Crear carpetas para guardar las imágenes
os.makedirs('/home/abner/Escritorio/omes', exist_ok=True)
os.makedirs('/home/abner/Escritorio/omes', exist_ok=True)

# Mapear las etiquetas de emociones
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

for index, row in data.iterrows():
    pixels = row['pixels']  # Cadena de píxeles
    emotion = row['emotion']  # Índice de emoción
    usage = row['Usage']  # Entrenamiento o prueba

    # Convertir los píxeles en una matriz 48x48
    pixels_array = np.array([int(pixel) for pixel in pixels.split()], dtype=np.uint8).reshape(48, 48)

    # Crear una imagen con PIL
    img = Image.fromarray(pixels_array)

    # Determinar la carpeta de destino
    if usage == 'Training':
        folder = 'images/train'
    else:
        folder = 'images/test'

    # Crear subcarpetas para cada emoción
    emotion_folder = os.path.join(folder, emotion_labels[emotion])
    os.makedirs(emotion_folder, exist_ok=True)

    # Guardar la imagen
    img_path = os.path.join(emotion_folder, f'{index}.png')
    img.save(img_path)

print("Imágenes convertidas y guardadas correctamente.")
