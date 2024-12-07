"""Sistema de reconocimiento automático de placas (QALPR).

Este script detecta vehículos, placas y accesorios en una imagen,
realiza OCR en las placasy genera una salida en formato JSON con la
información extraída.
"""
from typing import List, Dict, Any
from PIL import Image

from ocr_resnet.preprocessing import basic_image_processing
from qalpr_v2.inference_yolo import detect_obj_img
from qalpr_v2.metadata.doc_yolo_ocr import KeyObjDetections
from qalpr_v2.config import load_ocr_opt
from qalpr_v2.inference_ocr import ocr_placas


def qalpr(image: Image):
    """Proceso principal para el sistema de reconocimiento automático de placas (QALPR).
    
    Esta función realiza la detección de objetos en la imagen de entrada,
      asigna accesorios y placas,
    realiza OCR en las placas detectadas, y genera una salida formateada en JSON.

    Args:
        image: Imagen PIL cargada para procesar.
    
    Returns:
        Diccionario con el formato de salida.

    """
    detections = detect_obj_img(image)

    return detections


# pylint: disable=invalid-name
def process_ocr(detections: List[Dict[str, Any]], image_input: Image) -> List[str]:
    """Procesa una lista de detecciones de objetos para identificar placas,
    recortarlas y aplicar OCR.

    Args:
        detections: Lista de detecciones con clase, 
        coordenadas, y ID.
        image_input : Imagen PIL cargada.

    Returns:
        List: Lista de resultados de OCR para las placas detectadas.

    """
    placas_list = []
    OPT, MODEL, DEVICE, CONVERTER = load_ocr_opt()

    for detection in detections:
        if detection[KeyObjDetections.object_class] == 'placa':
            coordinates = detection[KeyObjDetections.coordinates]
            placa_id = detection[KeyObjDetections.id]

            cropped_image = basic_image_processing(image_input, coordinates)

            placas = ocr_placas(OPT, MODEL, DEVICE, CONVERTER, cropped_image, placa_id)
            placas_list.extend(placas)

    return placas_list


def process_ocr_codelab(image_input: Image) -> list[dict]:
    """Procesa una lista de detecciones de objetos para identificar placas,
    recortarlas y aplicar OCR.

    Args:
        detections: Lista de detecciones con clase,
        coordenadas, y ID.
        image_input : Imagen PIL cargada.

    Returns:
        List: Lista de resultados de OCR para las placas detectadas.

    """
    placas_list = []
    OPT, MODEL, DEVICE, CONVERTER = load_ocr_opt()

    cropped_image = image_input.convert('L')

    placas = ocr_placas(OPT, MODEL, DEVICE, CONVERTER, cropped_image, 1)
    placas_list.extend(placas)

    return placas_list
