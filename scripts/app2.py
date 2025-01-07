# scripts/app.py
import tkinter as tk
from tkinter import filedialog
from model import predict_emotion

# Función para cargar la imagen y mostrar la emoción
def upload_image():
    # Permitir al usuario seleccionar una imagen
    filepath = filedialog.askopenfilename()
    if filepath:
        # Predecir la emoción de la imagen seleccionada
        emotion = predict_emotion(filepath)
        label_result.config(text=f"Emotion: {emotion}")

# Crear la ventana principal de la aplicación
root = tk.Tk()
root.title("Emotion Detection")

# Etiqueta para la instrucción
label = tk.Label(root, text="Select an image")
label.pack()

# Botón para cargar la imagen
button_upload = tk.Button(root, text="Upload Image", command=upload_image)
button_upload.pack()

# Etiqueta para mostrar el resultado
label_result = tk.Label(root, text="")
label_result.pack()

# Ejecutar la interfaz
root.mainloop()
