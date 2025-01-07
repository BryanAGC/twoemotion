import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from preprocesamiento import cargar_datos, normalizar_datos
import matplotlib.pyplot as plt
# Cargar y preprocesar datos
ruta_csv = '/home/abner/Escritorio/fer/motion_detection/dataset/archive(3)/icml_face_data.csv'
x, y = cargar_datos(ruta_csv)
x = normalizar_datos(x)
y = to_categorical(y, num_classes=7)

# Dividir datos en entrenamiento y validación
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Definir el modelo
modelo = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Compilar el modelo
modelo.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Entrenar el modelo
historial = modelo.fit(x_train, y_train,
                       validation_data=(x_val, y_val),
                       epochs=25,
                       batch_size=64)

# Guardar el modelo entrenado
modelo.save('../modelos/modelo_emociones.h5')

# Graficar el historial de entrenamiento
plt.figure(figsize=(12, 6))
plt.plot(historial.history['accuracy'], label='Precisión Entrenamiento')
plt.plot(historial.history['val_accuracy'], label='Precisión Validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.grid()
plt.savefig('/home/abner/Escritorio/fer/motion_detection/resultados/grafica_entrenamiento.png')
plt.show()
