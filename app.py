from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def detect_emotion():
    uploaded_image = None  # Inicializa la variable para evitar el error
    emotion = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', emotion=None, uploaded_image=None)

        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            uploaded_image = filename  # Guarda el nombre del archivo subido

            try:
                # Procesar la imagen utilizando el archivo guardado
                image = cv2.imread(file_path)  # Usa cv2.imread en lugar de imdecode
                # Simula la detección de emociones (sustituye con tu lógica)
                emotion = "Happy" if "feliz" in filename else "Sad"  # Esto es solo un ejemplo
            except Exception as e:
                print(f"Error procesando la imagen: {e}")
                emotion = "Error procesando la imagen"

    return render_template('index.html', emotion=emotion, uploaded_image=uploaded_image)

if __name__ == '__main__':
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    
    # Usa el puerto proporcionado por Render o 5000 por defecto
    port = int(os.environ.get('PORT', 5000))
    # Ejecuta la aplicación en 0.0.0.0 para que sea accesible desde el exterior
    app.run(debug=True, host='0.0.0.0', port=port)
