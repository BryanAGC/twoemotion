# scripts/model.py
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Cargar el modelo previamente entrenado
model = load_model('/home/abner/Escritorio/fer/motion_detection/modelos/modelo_emociones.h5')

# Lista de emociones, según el dataset FER2013
emociones = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Función para predecir la emoción de una imagen
def predict_emotion(img_path):
    # Cargar la imagen y convertirla a formato adecuado
    img = image.load_img(img_path, target_size=(48, 48), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalizar la imagen
    
    # Hacer la predicción
    predictions = model.predict(img_array)
    predicted_emotion = emociones[np.argmax(predictions)]
    
    return predicted_emotion
