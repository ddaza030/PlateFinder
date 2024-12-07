"""Modulo para realizar deteccion con yolov8"""

# External libraries
import logging
from typing import List, Dict
import numpy as np
import cv2


from qalpr_v2.metadata.doc_yolo_ocr import KeyObjDetections
from qalpr_v2.config import load_yolo_model


def detect_obj_img(image: np.array) -> List[Dict]:
    """Realiza la inferencia en una imagen utilizando un modelo YOLO.

    Args:
        image: La ruta de la imagen en la que se realizará la inferencia.

    Returns:
        Una lista de diccionarios, donde cada diccionario contiene
        un ID único, las coordenadas de un objeto detectado, el tipo de objeto y
        la precisión/confianza de la detección.
        
    """

    logger = logging.getLogger(__name__)
    logger.info('Realizando la inferencia en la imagen')
    model = load_yolo_model()
    predict_model = model.predict(image, conf=0.59)

    detections = []
    detection_id = 1  # Inicializamos el contador para los IDs únicos
    for result in predict_model:
        for box in result.boxes:
            deteccion = {
                'id': detection_id,  # Asignamos el ID único
                KeyObjDetections.coordinates: box.xyxy[0].tolist(),
                KeyObjDetections.object_class: model.names[int(box.cls)],
                KeyObjDetections.presicion: box.conf.item(),
            }
            detections.append(deteccion)
            detection_id += 1  # Incrementamos el contador para la siguiente detección

    logger.info('Inferencia YOLO completada con éxito')
    return detections


def detect_obj_video(video_path: str) -> List[Dict]:
    """Realiza la inferencia en un video utilizando un modelo YOLO en batches de 40 frames.

    Args:
        video_path: Ruta del video a procesar.

    Returns:
        Una lista de diccionarios con la información de los objetos detectados,
        incluyendo su ubicación geográfica y el índice del cuadro.
    """
    logger = logging.getLogger(__name__)
    logger.info(f'Realizando la inferencia en el video: {video_path}')

    model = load_yolo_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"No se pudo abrir el video: {video_path}")
        return []

    detections = []
    frame_index = 0  # Índice del cuadro
    batch_size = 40  # Número de frames por batch
    frames_batch = []  # Almacenará los frames para el batch

    # Obtener el fps original del video
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    fps_target = 10  # fps deseado
    frame_interval = int(fps_original / fps_target)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            frames_batch.append(frame)  # Acumula el frame en el batch  # Acumula el frame en el batch

        # Si el batch está completo, realiza la inferencia
        if len(frames_batch) == batch_size:
            predict_model = model.predict(frames_batch, conf=0.59, verbose=False)

            # Procesa las predicciones por cada frame del batch
            for i, result in enumerate(predict_model):
                for box in result.boxes:
                    deteccion = {
                        KeyObjDetections.frames: frame_index + i,  # Índice del frame
                        KeyObjDetections.coordinates: box.xyxy[0].tolist(),
                        KeyObjDetections.presicion: box.conf.item(),
                        "imagen": frames_batch[i]  # Imagen correspondiente al frame
                    }
                    detections.append(deteccion)

            # Resetea el batch y actualiza el índice de frames
            frames_batch = []
            frame_index += batch_size

    # Si quedan frames que no fueron procesados por el batch
    if frames_batch:
        predict_model = model.predict(frames_batch, conf=0.59, verbose=False)
        for i, result in enumerate(predict_model):
            for box in result.boxes:
                deteccion = {
                    KeyObjDetections.frames: frame_index + i,
                    KeyObjDetections.coordinates: box.xyxy[0].tolist(),
                    KeyObjDetections.presicion: box.conf.item(),
                    "imagen": frames_batch[i]
                }
                detections.append(deteccion)

    cap.release()
    logger.info('Inferencia en el video completada con éxito')

    return detections