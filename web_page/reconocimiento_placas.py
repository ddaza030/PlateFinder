import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO
import requests
import tempfile

from qalpr_v2.qalpr import qalpr
from qalpr_v2.inference_yolo import detect_obj_video

# Lista de valores permitidos para reportes
REPORTES_VALIDOS = [
    "Cancelados",
    "Robados",
    "A nombre de persona indeterminada",
    "Identificados como prestadores de servicio de transporte informal",
    "Con solicitud de embargo",
    "Categorizados como contaminantes"
]

# Configuración del endpoint FastAPI
ENDPOINT_URL = "http://localhost:8000/procesar-imagen/"


def pagina_reconocimiento_placas():
    st.title("Sistema de Reconocimiento Automático de Placas (QALPR)")

    # Selección de fuente de datos
    option = st.radio(
        "Selecciona cómo deseas cargar los datos:",
        ("Subir imagen desde la galería", "Tomar foto con cámara", "Subir video"),
        horizontal=True
    )

    pil_image = None
    detections = []

    if option == "Tomar foto con cámara":
        image = st.camera_input("Captura una imagen para procesar")
        if image is not None:
            st.image(image, caption="Imagen capturada", use_container_width=True)
            pil_image = Image.open(image)

    elif option == "Subir imagen desde la galería":
        uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file)
            st.image(pil_image, caption="Imagen subida", use_container_width=True)

    elif option == "Subir video":
        video_file = st.file_uploader("Sube un video", type=["mp4", "avi", "mov"])
        if video_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(video_file.read())
                temp_video_path = temp_video.name

            placeholder = st.empty()
            placeholder.write("Procesando el video...")

            detections = detect_obj_video(temp_video_path)

            placeholder.empty()
            if detections:
                st.write(f"Se detectaron {len(detections)} objetos en el video.")

    # Procesar imágenes y enviar al endpoint si hay detecciones o una imagen cargada
    if pil_image is not None or detections:
        placeholder = st.empty()
        placeholder.write("Procesando la información...")

        data = []

        # Procesar imagen cargada
        if pil_image is not None:
            result = qalpr(pil_image)
            for placa in result:
                if placa["clase"] == "placa":
                    params = {
                        "latitud": "19.432608",
                        "longitud": "-99.133209",
                        "precision": placa["precision"],
                        "coordenadas": str(placa["coordenadas"]),
                    }
                    buffer = BytesIO()
                    pil_image.save(buffer, format="JPEG")
                    buffer.seek(0)
                    files = {"file": buffer}

                    response = requests.post(ENDPOINT_URL, params=params, files=files)
                    if response.status_code == 200:
                        api_response = response.json()
                        if not api_response["reportada"]:
                            estado = "No reportada"
                            color = "background-color: #d4edda; color: #155724;"  # Verde suave
                        else:
                            estado = ", ".join(api_response["reportes"])
                            color = "background-color: #f8d7da; color: #721c24;"  # Rojo suave

                        data.append({"Placa": api_response["placa"], "Estado": estado, "Color": color})

        # Procesar detecciones de video
        for det in detections:
            params = {
                "latitud": "19.432608",
                "longitud": "-99.133209",
                "precision": det["precision"],
                "coordenadas": str(det['coordenadas']),
            }
            pil_image = Image.fromarray(det["imagen"])
            buffer = BytesIO()
            pil_image.save(buffer, format="JPEG")
            buffer.seek(0)
            files = {"file": buffer}

            response = requests.post(ENDPOINT_URL, params=params, files=files)
            if response.status_code == 200:
                api_response = response.json()
                if not api_response["reportada"]:
                    estado = "No reportada"
                    color = "background-color: #d4edda; color: #155724;"  # Verde suave
                else:
                    estado = ", ".join(api_response["reportes"])
                    color = "background-color: #f8d7da; color: #721c24;"  # Rojo suave

                data.append({"Placa": api_response["placa"], "Estado": estado, "Color": color})

        placeholder.empty()
        if data:
            df = pd.DataFrame(data)

            def highlight_rows(row):
                return [row["Color"]] * len(row)

            st.write("Resultados:")
            st.dataframe(
                df.style.apply(highlight_rows, axis=1),
                use_container_width=True,
                column_order=["Placa", "Estado"]
            )
        else:
            st.write("No se detectaron placas.")
