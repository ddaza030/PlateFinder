import pandas as pd
from datetime import datetime
import os

# Ruta del archivo CSV
CSV_FILE = os.path.join("input", "vehiculos.csv")


# Función para cargar la base de datos
def load_database():
    """Carga la base de datos desde el archivo CSV o crea una nueva si no existe."""
    try:
        return pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        # Crear una base de datos vacía si el archivo no existe
        return pd.DataFrame(
            columns=["fecha_registro", "placa", "ubicacion_registro"])


# Función para guardar la base de datos
def save_database(df):
    """Guarda el DataFrame en el archivo CSV."""
    df.to_csv(CSV_FILE, index=False)


# Función para insertar un registro
def insert_record(placa, ubicacion_registro = None):
    """
    Inserta un nuevo registro en la base de datos.

    Args:
        placa (str): Placa del vehículo.
        ubicacion_registro (str): Ubicación donde se registró el vehículo.
    """
    df = load_database()
    nuevo_registro = {
        "fecha_registro": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "placa": [placa],
        "ubicacion_registro": [ubicacion_registro],
    }
    df = pd.concat([df, pd.DataFrame(nuevo_registro)], ignore_index=True)
    save_database(df)


# Función para consultar registros por placa
def get_records_by_placa(placa):
    """
    Obtiene todos los registros de una placa específica.

    Args:
        placa (str): Placa del vehículo.

    Returns:
        DataFrame: Registros correspondientes a la placa.
    """
    df = load_database()
    return df[df["placa"] == placa]


def existe_registro(placa):

    df = get_records_by_placa(placa)

    if not df.empty:
        existe = True
    else:
        existe = False

    return existe


# Función para filtrar registros por un rango de fechas
def get_records_by_date_range(start_date, end_date):
    """
    Filtra registros dentro de un rango de fechas.

    Args:
        start_date (str): Fecha inicial en formato "YYYY-MM-DD".
        end_date (str): Fecha final en formato "YYYY-MM-DD".

    Returns:
        DataFrame: Registros dentro del rango de fechas.
    """
    df = load_database()
    df["fecha_registro"] = pd.to_datetime(df["fecha_registro"])
    mask = (df["fecha_registro"] >= start_date) & (
                df["fecha_registro"] <= end_date)
    return df[mask]


# Función para eliminar registros por placa
def delete_records_by_placa(placa):
    """
    Elimina todos los registros de una placa específica.

    Args:
        placa (str): Placa del vehículo.
    """
    df = load_database()
    df = df[df["placa"] != placa]
    save_database(df)


# Función para obtener todas las placas registradas
def get_all_placas():
    """
    Obtiene una lista de todas las placas registradas en la base de datos.

    Returns:
        list: Lista de placas únicas.
    """
    df = load_database()
    return df["placa"].unique().tolist()


# Ejemplo de uso
if __name__ == "__main__":
    # Insertar registros
    insert_record("ABC445")
    insert_record("XYZ789")

