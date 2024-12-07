"""Codigo de asignacion de placas y vehiculo segun el ID correspondiente"""

from typing import List, Dict, Any


def assign_plates_to_vehicles(
    assignment: Dict[str, List[Dict[str, Any]]], placas_list: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Asigna los datos de OCR de las placas a los vehículos en el
      diccionario de asignación.

    Args:
        assignment : Diccionario que contiene tipos
          de vehículos y su información.
        placas_list : Lista de datos de OCR de placas, con ID y
          datos de confianza.

    Returns:
        Diccionario `assignment` actualizado con las
          placas asignadas a los vehículos.

    """
    for _, vehiculos in assignment.items():
        for vehiculo in vehiculos:
            placa_asignada = vehiculo.get('placa')

            if placa_asignada:
                placa_id = placa_asignada.get('id')

                for placa_data in placas_list:
                    if placa_data['id'] == placa_id:
                        vehiculo['placa']['numero'] = placa_data['placa']
                        vehiculo['placa']['confianza_ocr'] = placa_data['confianza']
                        break

    return assignment
