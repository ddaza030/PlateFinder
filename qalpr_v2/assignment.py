"""Este conjunto de funciones está diseñado para realizar la asignación de placas
y accesorios a vehículos, utilizando una métrica de penalización basada en la
distancia geométrica entre los centros de las bounding boxes de las detecciones.
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from qalpr_v2.config import lists_assignament, load_config_assignment
from qalpr_v2.metadata.doc_yolo_ocr import KeyAssignment, KeyObjDetections


def calculate_bbox_center(coordenadas: List[int]) -> Tuple[int, int]:
    """Retorna el centro geométrico de una bounding box dada por
    una lista de coordenadas [left, top, right, bottom].

    Args:
        coordenadas: Lista con las coordenadas de la bounding box
                     en el formato [left, top, right, bottom].

    Returns:
        Coordenadas del centro de la bounding box (x, y).

    """
    left, top, right, bottom = coordenadas
    center_x = int((right - left) / 2 + left)
    center_y = int((bottom - top) / 2 + top)

    return center_x, center_y


def create_penalty_matrix(
    detected_vhls: List[dict], detected_accesories: List[dict]
) -> np.ndarray:
    """Crea una matriz de penalización basada en la distancia entre
    los centros geométricos de dos listas de bounding boxes.

    Args:
        detected_vhls: Lista de diccionarios que contienen las coordenadas de los
                 vehículos o accesorios.
        detected_accesories: Lista de diccionarios que contienen las coordenadas de las
                placas o accesorios a comparar.

    Return:
        Matriz de penalización que representa el costo de asignar cada elemento
        de `detected_vhls` a cada elemento de `detected_accesories`.

    """
    center_vhls = np.asarray(
        [calculate_bbox_center(a[KeyObjDetections.coordinates]) for a in detected_vhls]
    )
    center_accesories = np.asarray(
        [
            calculate_bbox_center(b[KeyObjDetections.coordinates])
            for b in detected_accesories
        ]
    )
    penalty_matrix = np.sum((center_vhls[:, np.newaxis] - center_accesories) ** 2, axis=2)

    return penalty_matrix


def match_plates(
    cost_matrix: np.ndarray, allowed_cost: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Asigna placas a vehículos utilizando el algoritmo de asignación de suma mínima.

    Args:
        cost_matrix: Matriz de penalización que define los costos de asignación.
        allowed_cost: Costo máximo permitido para realizar una asignación.

    Return:
        Lista de pares (vehículo, placa), lista de vehículos huérfanos,
        y lista de placas huérfanas.

    """
    pairs = []
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    v_orphan = [i for i in range(cost_matrix.shape[0]) if i not in row_indices]
    p_orphan = [i for i in range(cost_matrix.shape[1]) if i not in col_indices]

    for row_idx, col_idx in zip(row_indices, col_indices):
        if cost_matrix[row_idx, col_idx] <= allowed_cost:
            pairs.append((row_idx, col_idx))
        else:
            v_orphan.append(row_idx)
            p_orphan.append(col_idx)

    return pairs, v_orphan, p_orphan


def catalog_detections(detections: List[Dict]) -> Dict[str, List]:
    """Clasifica las detecciones en diferentes categorías (vehículos, placas,
    accesorios de motos y accesorios de vehículos).

    Args:
        detections: Lista de detecciones, donde cada detección es un diccionario
        con las claves definidas en `KeyObjDetections`.

    Return:
        Diccionario que contiene las detecciones clasificadas en diferentes categorías.
    """
    output = {
        KeyAssignment.plates: [],
        KeyAssignment.vhls: [],
        KeyAssignment.bike_accesories: [],
        KeyAssignment.vhl_accesories: [],
    }

    vehicle_classes, plates_classes, bike_acces_classes, vhl_acces_classes = (
        lists_assignament()
    )

    for detection in detections:
        label = detection[KeyObjDetections.object_class]

        detection_info = {
            KeyObjDetections.id: detection[KeyObjDetections.id],
            'confianza': detection[KeyObjDetections.presicion],
            'label': detection[KeyObjDetections.object_class],
            'coordenadas': detection[KeyObjDetections.coordinates],
        }

        if label in plates_classes:
            output[KeyAssignment.plates].append(detection_info)
        elif label in vehicle_classes:
            output[KeyAssignment.vhls].append(detection_info)
        elif label in bike_acces_classes:
            output[KeyAssignment.bike_accesories].append(detection_info)
        elif label in vhl_acces_classes:
            output[KeyAssignment.vhl_accesories].append(detection_info)

    return output


def assign_accessories(
    motos: List[Dict],
    otros_vehiculos: List[Dict],
    bike_accessories: List[Dict],
    vhl_acces: List[Dict],
    costo_permitido: float,
) -> Tuple[List[List[Dict]], List[List[Dict]]]:
    """Asigna accesorios a motocicletas y otros vehículos.

    Args:
        motos: Lista de motocicletas.
        otros_vehiculos: Lista de otros vehículos.
        bike_accessories: Lista de accesorios de motocicletas.
        vhl_acces: Lista de accesorios de otros vehículos.
        costo_permitido: Costo máximo permitido para asignar accesorios.

    Return:
        Lista de listas de accesorios asignados a motocicletas y otros vehículos.
        Cada elemento de las listas es una lista que contiene los accesorios
        asignados a cada vehículo o motocicleta.
    """
    assigned_bike_acces = [[] for _ in motos]
    assigned_vhl_acces = [[] for _ in otros_vehiculos]

    # Asignar accesorios a motocicletas
    if motos and bike_accessories:
        bike_accessories_penalization_matrix = create_penalty_matrix(
            motos, bike_accessories
        )
        assigned_accessories = set()

        for i, moto in enumerate(motos):
            for j, accessories in enumerate(bike_accessories):
                if (
                    j not in assigned_accessories
                    and bike_accessories_penalization_matrix[i, j] <= costo_permitido
                ):
                    assigned_bike_acces[i].append(accessories)
                    assigned_accessories.add(j)

    if otros_vehiculos and vhl_acces:
        matrix_penalization_accessories_veh = create_penalty_matrix(
            otros_vehiculos, vhl_acces
        )
        assigned_accessories = set()

        for i, vehicle in enumerate(otros_vehiculos):
            for j, accessories in enumerate(vhl_acces):
                if (
                    j not in assigned_accessories
                    and matrix_penalization_accessories_veh[i, j] <= costo_permitido
                ):
                    assigned_vhl_acces[i].append(accessories)
                    assigned_accessories.add(j)

    return assigned_bike_acces, assigned_vhl_acces


def assign_accessories_and_plates(
    detections: List[Dict]
) -> Dict[str, List]:
    """Asigna placas y accesorios a vehículos utilizando la distancia entre ellos
    y un costo permitido.

    Args:
        detections: Lista de detecciones de objetos.
        costo_permitido: Costo máximo permitido para asignar accesorios
                         o placas a vehículos.

    Return:
        dict: Diccionario con asignaciones de vehículos, placas y accesorios
              organizados en 'bikes' y 'others_vhls'.
    """
    catalog = catalog_detections(detections)

    # Si la penalización (o distancia) entre el vehículo y
    # la placa es menor o igual a costo_permitido, la asignación se considera válida.
    costo_permitido = load_config_assignment()

    assignments = {KeyAssignment.bikes: [], KeyAssignment.others_vhls: []}

    v_orphan = []

    if catalog[KeyAssignment.vhls] and catalog[KeyAssignment.plates]:
        couples_plates, v_orphan, p_orphan = match_plates(
            create_penalty_matrix(
                catalog[KeyAssignment.vhls], catalog[KeyAssignment.plates]
            ),
            costo_permitido,
        )

        for v_indx, p_indx in couples_plates:
            vehicle = catalog[KeyAssignment.vhls][v_indx]
            plate_ass = catalog[KeyAssignment.plates][p_indx]

            if vehicle[KeyAssignment.label] == KeyAssignment.one_bike:
                assignments[KeyAssignment.bikes].append(
                    {
                        KeyAssignment.vehicle: vehicle,
                        KeyAssignment.plate: plate_ass,
                        KeyAssignment.accessories: [],
                    }
                )
            else:
                assignments[KeyAssignment.others_vhls].append(
                    {
                        KeyAssignment.vehicle: vehicle,
                        KeyAssignment.plate: plate_ass,
                        KeyAssignment.accessories: [],
                    }
                )

    for v_indx in v_orphan:
        if (
            catalog[KeyAssignment.vhls][v_indx][KeyAssignment.label]
            == KeyAssignment.one_bike
        ):
            assignments[KeyAssignment.bikes].append(
                {
                    KeyAssignment.vehicle: catalog[KeyAssignment.vhls][v_indx],
                    KeyAssignment.plate: None,
                    KeyAssignment.accessories: [],
                }
            )
        else:
            assignments[KeyAssignment.others_vhls].append(
                {
                    KeyAssignment.vehicle: catalog[KeyAssignment.vhls][v_indx],
                    KeyAssignment.plate: None,
                    KeyAssignment.accessories: [],
                }
            )

    listados = {
        KeyAssignment.bikes: [
            moto[KeyAssignment.vehicle] for moto in assignments[KeyAssignment.bikes]
        ],
        KeyAssignment.others_vhls: [
            auto[KeyAssignment.vehicle] for auto in assignments[KeyAssignment.others_vhls]
        ],
    }

    # Se utiliza costo_permitido para asignar accesorios de motocicletas
    # y vehículos basándose en la penalización calculada entre ellos.
    assigned_accessories = assign_accessories(
        listados[KeyAssignment.bikes],
        listados[KeyAssignment.others_vhls],
        catalog[KeyAssignment.bike_accesories],
        catalog[KeyAssignment.vhl_accesories],
        costo_permitido,
    )

    # Iterar sobre cada motocicleta asignada en el diccionario `assignments`
    for i, vhc in enumerate(assignments[KeyAssignment.bikes]):
        # Verificar si el índice actual tiene un conjunto de accesorios asignados
        #  en `assigned_accessories[0]`
        if i < len(assigned_accessories[0]):
            # Si es así, agregar los accesorios correspondientes a la motocicleta actual
            vhc[KeyAssignment.accessories].extend(assigned_accessories[0][i])

    # Iterar sobre cada vehículo (no motocicleta) asignado en el diccionario `assignments`
    for i, vhc in enumerate(assignments[KeyAssignment.others_vhls]):
        # Verificar si el índice actual tiene un conjunto de accesorios asignados
        #  en `assigned_accessories[1]`
        if i < len(assigned_accessories[1]):
            # Si es así, agregar los accesorios correspondientes al vehículo actual
            vhc[KeyAssignment.accessories].extend(assigned_accessories[1][i])

    return assignments
