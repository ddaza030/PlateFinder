"""Documentación de la inferencia por modelo YOLOv8 y OCR"""


class KeyObjDetections:
    """Lista de diccionarios que contiene: ID de la detección, las coordenadas,
      tipo de objeto,y precisión de las predicciones de un modelo en cada imagen."""

    coordinates = 'coordenadas'
    """Coordenadas del objeto detectado en la imagen, almacenadas en 
    formato [x1, y1, x2, y2], donde:
    - x1, y1: esquina superior izquierda
    - x2, y2: esquina inferior derecha
    """

    object_class = 'clase'
    """Clase del objeto detectado, representada por su nombre según la etiqueta (str)"""

    presicion = 'precision'
    """Precisión o score de confianza de la predicción del objeto (float)"""

    latitud = 'latitud'
    """Coordenada latitud de donde se corrio el codigo (float)"""

    longitud = 'longitud'
    """Coordenada longitud de donde se corrio el codigo (float)"""

    frames = 'frame'
    """frames cargados por video"""

class KeyOCR:
    """Lista que contiene: id de la deteccion, texto extraido
    de la placa, y confianza de la prediccion del modelo OCR"""

    id = 'id'
    """ID de la deteccion (int)"""

    texto = 'placa'
    """Texto extraido de la placa por OCR (str)"""

    confianza = 'confianza'
    """Precisión o score de confianza de la predicción de la placa (float)"""


class KeyAssignment:
    """Listas: placas, vehiculos, accesorios divididos y vehiculos divididos"""

    plates = 'placas'
    """Placas detectadas en la imagen [List]"""

    vhls = 'Vehiculos'
    """Vehiculos detectados en la imagen [List]"""

    bike_accesories = 'Accesorios de motos'
    """Accesorios de moto (Cascos) detectados en la imagen [List]"""

    vhl_accesories = 'Accesorios de vehiculos diferente a moto'
    """Accesorios de vehiculos que no sean moto (Cinturones)
      detectados en la imagen [List]"""

    bikes = 'Motos'
    """Motos detectadas en la imagen [List]"""

    others_vhls = 'Vehiculos diferentes a moto'
    """Vehiculos que no son motos detectados en la imagen [List]"""

    vehicle = 'vehiculo'
    """Vehiculos detectados"""

    plate = 'placa'
    """Placa detectada"""

    accessories = 'accesorios'
    """Accesorios detectados"""

    label = 'label'
    """Etiqueta asociada a un tipo de vehiculo"""

    one_bike = 'moto'
    """Valor de label que indica que es una motocicleta"""
