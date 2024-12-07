""""Modulo para leer y cargar pesos."""

import argparse
from functools import lru_cache
from typing import List, Tuple
import yaml

import torch
from torch.backends import cudnn
from ultralytics import YOLO

from ocr_resnet.utils import AttnLabelConverter
from ocr_resnet.model import Model
from qalpr_v2.metadata.path import Path


def dict_to_namespace(config_dict):
    """Función para convertir un diccionario en un argparse.Namespace."""
    for key, value in config_dict.items():
        if isinstance(value, dict):
            config_dict[key] = dict_to_namespace(value)
    return argparse.Namespace(**config_dict)


@lru_cache()
def load_config(yaml_file):
    """Cargar argumentos de configuración para modelo OCR."""
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


@lru_cache()
def load_yolo_model():
    """"Cargar modelo de Yolo para 
    la deteccion de objetos en imagenes."""

    model = YOLO(Path.yolo_model)

    return model


@lru_cache()
def load_ocr_opt():
    """"Cargar modelo y configuraciones de OCR para 
    la lectura de placas."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = load_config(Path.config_ocr)
    opt = dict_to_namespace(config)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    model = Model(opt)
    mode_org = torch.nn.DataParallel(model).to(device)
    mode_org.load_state_dict(torch.load(Path.ocr_model, map_location=device))

    return opt, mode_org, device, converter


@lru_cache()
def load_dict_assignment(yaml_file: str) -> dict:
    """Carga un archivo YAML que contiene la asignación de clases.

    Args:
        yaml_file: Ruta del archivo YAML.

    Return:
        dict: Diccionario con la asignación de clases.

    """
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)

    return data


@lru_cache()
def lists_assignament() -> Tuple[List, List, List, List]:
    """Carga y retorna las listas de clases relacionadas con vehículos,
    placas, accesorios de motos y accesorios de vehículos desde un archivo
    de configuración YAML.

    Args:
         Ruta del archivo YAML donde están definidas las clases.

    Return:
        Cuatro listas que contienen las clases de vehículos, placas,
        accesorios de motos y accesorios de vehículos, respectivamente.

    """
    data = load_dict_assignment(Path.config_assignament)
    vehicle_classes = data.get('clases_vehiculo')
    plates_classes = data.get('clases_placas')
    bike_acces_classes = data.get('clases_accesorios_moto')
    vhl_acces_classes = data.get('clases_accesorios_vhls')

    return (vehicle_classes, plates_classes, bike_acces_classes, vhl_acces_classes)


@lru_cache()
def load_config_assignment() -> int:
    """Carga un archivo YAML y devuelve el valor de 'costo_permitido'.

    Args:
        yaml_file (str): Ruta del archivo YAML.

    Returns:
        int: El valor de 'costo_permitido' en el archivo YAML.
    """
    yaml_file = Path.config_meta_assignament
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)

    return data.get('costo_permitido', None)
