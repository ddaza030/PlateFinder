import streamlit as st
from pymongo import MongoClient
import pandas as pd
from PIL import Image
from io import BytesIO
import requests

# Importar función QALPR
from qalpr_v2.qalpr import qalpr  # Asegúrate de que esta ruta sea correcta
from utils.bd_mongo_reportes import insert_or_update_record
# from web_page.reconocimiento_placas import pagina_reconocimiento_placas

# Configuración de MongoDB
MONGO_URI = "mongodb://admin:admin@localhost:27017"
DATABASE_NAME = "vehiculos_db"
COLLECTION_NAME = "vehiculos"

# Conexión con MongoDB
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

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


# Función para la primera página: Reconocimiento de placas
def pagina_reconocimiento_placas():
    st.title("Sistema de Reconocimiento Automático de Placas (QALPR)")

    # Selección de fuente de imagen
    option = st.radio(
        "Selecciona cómo deseas cargar la imagen:",
        ("Subir imagen desde la galería", "Tomar foto con cámara"),
        horizontal=True
    )

    pil_image = None

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

    # Procesar y enviar datos si hay una imagen cargada
    if pil_image is not None:
        placeholder = st.empty()
        placeholder.write("Procesando la imagen...")

        result = qalpr(pil_image)
        data = []

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
                        # Placa no reportada
                        estado = "No reportada"
                        color = "background-color: #d4edda; color: #155724;"  # Verde suave
                    else:
                        # Placa reportada, listar los reportes
                        estado = ", ".join(api_response["reportes"])
                        color = "background-color: #f8d7da; color: #721c24;"  # Rojo suave

                    data.append({"Placa": api_response["placa"], "Estado": estado, "Color": color})
                else:
                    st.error(f"Error al procesar la placa ID {placa['id']}: {response.text}")

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
            st.write("No se detectó placa")


# Función para la segunda página: Gestión de reportes
def pagina_gestion_reportes():
    st.title('Registro de Vehículos y Reportes')

    # Consultar los vehículos
    vehiculos = collection.find({}, {"_id": 0, "placa": 1, "reportes": 1})
    df = pd.DataFrame(vehiculos)

    if not df.empty:
        st.write("Lista de vehículos registrados y sus reportes:")
        column_to_filter = st.radio("Seleccionar columna para filtrar", df.columns)
        filter_value = st.text_input(f"Ingrese valor para filtrar en la columna {column_to_filter}")
        st.dataframe(df)

        if filter_value:
            filtered_df = df[df[column_to_filter].str.contains(filter_value, case=False, na=False)]
            st.write("Resultados del filtro:")
            st.dataframe(filtered_df)
    else:
        st.write("No hay vehículos registrados.")

    placa_input = st.text_input("Placa del vehículo Insertar o Actualizar")

    if placa_input:
        existing_record = collection.find_one({"placa": placa_input})

        if existing_record:
            st.write(f"Placa {placa_input} ya registrada. ¿Deseas actualizar los reportes?")
            current_reportes = existing_record.get("reportes", [])
            st.write(f"Reportes actuales: {', '.join(current_reportes)}")
            reportes_input = st.multiselect(
                "Selecciona los reportes (puedes elegir varios)",
                REPORTES_VALIDOS, default=current_reportes
            )
        else:
            st.write(f"Placa {placa_input} no encontrada. Puedes agregar nuevos reportes.")
            reportes_input = st.multiselect(
                "Selecciona los reportes (puedes elegir varios)",
                REPORTES_VALIDOS
            )
    else:
        reportes_input = []

    if st.button("Insertar o actualizar registro"):
        if placa_input and reportes_input:
            insert_or_update_record(placa_input, reportes_input)
            st.success(f"Registro actualizado para la placa {placa_input}")
        else:
            st.error("Debe ingresar una placa y seleccionar al menos un reporte.")


# Sidebar para la navegación
st.sidebar.title("Menú")
opcion = st.sidebar.radio("Selecciona una opción:", ["Reconocimiento de Placas", "Gestión de Reportes"])

if opcion == "Reconocimiento de Placas":
    pagina_reconocimiento_placas()
elif opcion == "Gestión de Reportes":
    pagina_gestion_reportes()
