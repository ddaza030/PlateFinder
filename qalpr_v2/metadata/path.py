"""" Modulo para localizar modelo"""

import os


class Path:
    """Relevant directories of the project."""

    folder = os.getcwd()

    _input = os.path.join(folder, 'input')
    """Path to the input folder."""

    _weights = os.path.join(_input, 'weights')
    """Path to the input folder."""

    yolo_model = os.path.join(_weights, 'best.pt')
    """Path to the Yolo weights."""

    ocr_model = os.path.join(_weights, 'ocr_v3.pth')
    """Path to the OCR weights."""

    _configs = os.path.join(_input, 'configs')
    """Path to configs"""

    config_ocr = os.path.join(_configs, 'config_ocr.yaml')
    """Path to configs"""

    config_assignament = os.path.join(_configs, 'groups_assignment.yaml')
    """Path to list of assignaments"""

    config_meta_assignament = os.path.join(_configs, 'assignment.yaml')
    """Path to metadata and config of assignaments"""
    