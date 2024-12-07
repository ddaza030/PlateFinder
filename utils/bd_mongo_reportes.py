from pymongo import MongoClient, ASCENDING
from datetime import datetime
import random
import time

# Configuración de MongoDB
MONGO_URI = "mongodb://admin:admin@localhost:27017"
DATABASE_NAME = "vehiculos_db"
COLLECTION_NAME = "vehiculos"

# Conexión con MongoDB
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Crear índice único en 'placa' (si no existe)
collection.create_index([("placa", ASCENDING)], unique=True)

# Lista de valores permitidos para reportes
REPORTES_VALIDOS = [
    "Cancelados",
    "Robados",
    "A nombre de persona indeterminada",
    "Identificados como prestadores de servicio de transporte informal",
    "Con solicitud de embargo",
    "Categorizados como contaminantes"
]

# Función para insertar o actualizar un registro
def insert_or_update_record(placa, reportes=None):
    """
    Inserta un nuevo registro o actualiza un existente en la colección.

    Args:
        placa (str): Placa del vehículo.
        reportes (list): Lista de reportes a asociar con la placa.
    """
    if not reportes:
        reportes = []

    # Validar que los reportes sean válidos
    reportes_validos = [reporte for reporte in reportes if reporte in REPORTES_VALIDOS]

    if not reportes_validos:
        print(f"No se añadieron reportes inválidos para la placa: {placa}")

    fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    collection.update_one(
        {"placa": placa},
        {
            "$set": {"fecha_ultimo_reporte": fecha_actual},
            "$addToSet": {"reportes": {"$each": reportes_validos}},
        },
        upsert=True
    )


# Función para obtener los datos de una placa
def get_record_by_placa(placa):
    """
    Obtiene el registro de una placa específica.

    Args:
        placa (str): Placa del vehículo.

    Returns:
        dict: Registro correspondiente a la placa.
    """
    return collection.find_one({"placa": placa}, {"_id": 0})


# Función para eliminar registros por placa
def delete_records_by_placa(placa):
    """
    Elimina un registro de una placa específica.

    Args:
        placa (str): Placa del vehículo.
    """
    collection.delete_one({"placa": placa})


# Función para obtener todas las placas registradas
def get_all_placas():
    """
    Obtiene una lista de todas las placas registradas en la colección.

    Returns:
        list: Lista de placas únicas.
    """
    return collection.distinct("placa")


def get_reportes_by_placa(placa):
    """
    Obtiene la lista de reportes asociados a una placa específica.

    Args:
        placa (str): Placa del vehículo.

    Returns:
        list: Lista de reportes asociados a la placa. Si no se encuentra, retorna una lista vacía.
    """
    registro = collection.find_one({"placa": placa}, {"_id": 0, "reportes": 1})
    return registro.get("reportes", []) if registro else []


# Función para insertar múltiples registros en un solo insert
def bulk_insert_records(start, end):
    registros = []

    for i in range(start, end):
        # Generar una placa única (por ejemplo, formato AAA123)
        placa = f"{chr(65 + i % 26)}{chr(65 + (i // 26) % 26)}{chr(65 + (i // 676) % 26)}{i % 1000:03d}"

        # Seleccionar reportes aleatorios (pueden ser de 0 a 3 reportes)
        reportes_aleatorios = random.sample(REPORTES_VALIDOS,
                                            random.randint(1, 3))

        # Crear el registro para este documento
        registro = {
            "placa": placa,
            "reportes": reportes_aleatorios,
            "fecha_registro": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Agregar el registro a la lista
        registros.append(registro)

    # Insertar todos los registros de una vez
    if registros:
        collection.insert_many(registros)


# Ejemplo de uso
if __name__ == "__main__":
    insert_or_update_record("HZE659", ["Robados", "Con solicitud de embargo"])
    insert_or_update_record("HMG527", ["Robados"])
    insert_or_update_record("XYZ789", ["Categorizados como contaminantes"])
    insert_or_update_record("ABC445", ["Cancelados"])  # Agregar otro reporte a 'ABC445'

    bulk_insert_records(200000, 300000)
    # Medir tiempo para consultar el registro completo

    start_time = time.time()
    reportes = get_reportes_by_placa("ABC445")
    end_time = time.time()
    print(
        f"Tiempo de para encontrar una placa entre {collection.count_documents({})}:"
        f" {end_time - start_time:.6f} segundos",
    )
