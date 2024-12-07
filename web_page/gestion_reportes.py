import streamlit as st
import pandas as pd
from utils.bd_mongo_reportes import insert_or_update_record
from pymongo import MongoClient

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