import pandas as pd
import numpy as np

def cargar_datos(ruta_csv):
    """
    """
    datos = pd.read_csv(ruta_csv)
    
    # Convertir la columna de píxeles a arrays numéricos
    pixeles = datos['pixels'].apply(lambda x: np.fromstring(x, sep=' '))
    
    # Filtrar solo las filas con exactamente 2304 valores (48x48)
    pixeles = pixeles[pixeles.apply(lambda x: len(x) == 2304)]
    etiquetas = datos.loc[pixeles.index, 'emotion'].values
    
    # Reshape de los píxeles a (48, 48)
    pixeles = np.stack(pixeles.apply(lambda x: x.reshape(48, 48)))
    return pixeles, etiquetas

def normalizar_datos(x):
    """
    Normaliza los datos de entrada (imágenes) entre 0 y 1.
    """
    x = x / 255.0  # Escalar valores entre 0 y 1
    x = x.reshape(-1, 48, 48, 1)  # Agregar dimensión para canal de color (grises)
    return x
