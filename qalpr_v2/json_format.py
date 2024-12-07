"""Este script genera una salida en formato JSON a partir de un diccionario que contiene
la asignación de vehículos y sus accesorios. Para cada vehículo detectado, se incluyen
detalles sobre el tipo de vehículo, coordenadas, nivel de confianza, y si aplica, la
información de la placa y los accesorios detectados (como cascos o cinturones).
"""

import json


def create_json_output(assignment, image_name):
    """Genera un JSON con las detecciones de vehículos y accesorios.

    Args:
        assignment (dict): Diccionario que contiene los datos de detección de vehículos,
                        placas y accesorios.
        image_name (str): Nombre de la imagen procesada.

    Returns:
        str: Cadena en formato JSON con los detalles de los vehículos, placas.
    """
    output = {"nombre_Imagen": image_name, "estado": "Ok", "detecciones": []}

    for vehicle_type, vehicles in assignment.items():
        for vehicle_data in vehicles:
            vehicle = vehicle_data.get('vehiculo', {}) or {}
            plate = vehicle_data.get('placa', {}) or {}
            accessories = vehicle_data.get('accesorios', []) or []

            detection = {
                "clase_vehiculo": vehicle.get('label', 'N/A'),
                "score_vehiculo": f"{vehicle.get('confianza', 0) * 100:.2f}",
                "coord_vehiculo": [
                    int(coord) for coord in vehicle.get('coordenadas', [])
                ],
                "info_placa": {
                    "placa": plate.get('numero', 'N/A'),
                    "score_ocr_placa": plate.get('confianza_ocr', 'N/A'),
                    "score_placa": f"{plate.get('confianza', 0) * 100:.2f}",
                    "coord._placa": [
                        int(coord) for coord in plate.get('coordenadas', [])
                    ],
                },
                "accesorios_vehiculo": {},
            }

            if vehicle.get('label') == 'moto':
                info_cascos = {
                    "Numero_cascos": 0,
                    "Numero_no_cascos": 0,
                    "coord_cascos": [],
                    "coord_no_cascos": [],
                }

                for accessory in accessories:
                    if accessory.get('label') == 'casco':
                        info_cascos["Numero_cascos"] += 1
                        info_cascos["coord_cascos"].append(
                            [int(coord) for coord in accessory.get('coordenadas', [])]
                        )
                    else:
                        info_cascos["Numero_no_cascos"] += 1
                        info_cascos["coord_no_cascos"].append(
                            [int(coord) for coord in accessory.get('coordenadas', [])]
                        )

                detection["accesorios_vehiculo"] = {"Info_cascos": info_cascos}

            else:
                info_cinturones = {
                    "cinturon_conductor": 0,
                    "coord_cinturon_conductor": [],
                    "cinturon_copiloto": 0,
                    "coord_cinturon_copiloto": [],
                    "no_cinturon": 0,
                }

                for accessory in accessories:
                    if accessory.get('label') == 'cinturon_conductor':
                        info_cinturones["cinturon_conductor"] += 1
                        info_cinturones["coord_cinturon_conductor"].append(
                            [int(coord) for coord in accessory.get('coordenadas', [])]
                        )
                    elif accessory.get('label') == 'cinturon_copiloto':
                        info_cinturones["cinturon_copiloto"] += 1
                        info_cinturones["coord_cinturon_copiloto"].append(
                            [int(coord) for coord in accessory.get('coordenadas', [])]
                        )
                    else:
                        info_cinturones["no_cinturon"] += 1

                detection["accesorios_vehiculo"] = {"Info_cinturones": info_cinturones}

            output["detecciones"].append(detection)

    return output
