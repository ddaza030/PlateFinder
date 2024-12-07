from pymongo import MongoClient, ASCENDING
from datetime import datetime

# Configuración de MongoDB
MONGO_URI = "mongodb://admin:admin@localhost:27017"
DATABASE_NAME = "vehiculos_db"
COLLECTION_NAME = "vehiculos_maestro"

# Conexión con MongoDB
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Crear índice compuesto único en 'placa' y 'fecha_hora' (si no existe)
collection.create_index([("placa", ASCENDING), ("fecha_hora", ASCENDING)],
                        unique=True)


# Función para insertar o actualizar un registro maestro
def insert_or_update_record_maestro(placa, foto, geolocalizacion,
                                    resultado_listas):
    """
    Inserta un nuevo registro maestro o actualiza uno existente.

    Args:
        placa (str): Placa del vehículo.
        foto (str): Ruta o referencia de la foto de la placa.
        geolocalizacion (dict): Diccionario con 'latitud' y 'longitud'.
        resultado_listas (list): Resultados de la presencia en listas de interés.
    """
    fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    collection.update_one(
        {"placa": placa, "fecha_hora": fecha_hora},
        {
            "$set": {
                "foto": foto,
                "geolocalizacion": geolocalizacion,
                "resultado_listas": resultado_listas
            }
        },
        upsert=True
    )


# Función para obtener un registro maestro por placa y fecha_hora
def get_record_by_placa_maestro(placa, fecha_hora):
    """
    Obtiene el registro maestro de una placa específica y fecha_hora.

    Args:
        placa (str): Placa del vehículo.
        fecha_hora (str): Fecha y hora del registro.

    Returns:
        dict: Registro maestro correspondiente a la placa y fecha_hora.
    """
    return collection.find_one({"placa": placa, "fecha_hora": fecha_hora},
                               {"_id": 0})


# Función para eliminar registros por placa y fecha_hora
def delete_record_by_placa_maestro(placa, fecha_hora):
    """
    Elimina un registro maestro de una placa específica y fecha_hora.

    Args:
        placa (str): Placa del vehículo.
        fecha_hora (str): Fecha y hora del registro.
    """
    collection.delete_one({"placa": placa, "fecha_hora": fecha_hora})


# Función para obtener todas las placas registradas
def get_all_placas_maestro():
    """
    Obtiene una lista de todas las placas registradas en la colección maestro.

    Returns:
        list: Lista de placas únicas.
    """
    return collection.distinct("placa")


# Función para obtener todos los registros de una placa, sin necesidad de fecha_hora
def get_by_placa(placa):
    """
    Obtiene todos los registros maestros de una placa.

    Args:
        placa (str): Placa del vehículo.

    Returns:
        list: Lista de registros para la placa especificada.
    """
    return list(collection.find({"placa": placa}, {"_id": 0}))


def test_insert_or_update_record():
    """
    Función para probar la inserción o actualización de un registro en la colección maestro.
    """
    # Valores de prueba
    placa = "ABC123"
    foto = "ruta/a/la/foto.jpg"
    geolocalizacion = {"latitud": "19.4326", "longitud": "-99.1332"}
    resultado_listas = ["lista1", "lista2"]

    # Insertar o actualizar el registro
    insert_or_update_record_maestro(placa, foto, geolocalizacion, resultado_listas)

    # Verificar si el registro se insertó correctamente
    fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    registro = get_record_by_placa_maestro(placa, fecha_hora)

    if registro:
        print("Registro insertado correctamente:", registro)
    else:
        print("No se encontró el registro.")

    # Obtener todos los registros de la placa "ABC123"
    registros_placa = get_by_placa(placa)
    if registros_placa:
        print("Registros de la placa:", registros_placa)
    else:
        print("No se encontraron registros para la placa.")


if __name__ == "__main__":
    test_insert_or_update_record()