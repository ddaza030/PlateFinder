"""Este código contiene varias funciones para la manipulación y normalización
de imágenes, en particular para su uso en modelos de reconocimiento de texto (OCR).
"""

# pylint: disable=C0103
import math
import os
from typing import List

import torch
from torchvision import transforms
from PIL import Image
from natsort import natsorted


def resize_normalize(img, size, interpolation=Image.BICUBIC):
    """Cambia el tamaño de la imagen y la normaliza.
    Args:
        img (PIL.Image): Imagen de entrada.
        size (tuple): Nuevo tamaño de la imagen.
        interpolation (PIL.Image): Tipo de interpolación para el redimensionado.

    Returns:
        torch.Tensor: Imagen redimensionada y normalizada.
    """
    to_tensor = transforms.ToTensor()
    img = img.resize(size, interpolation)
    img = to_tensor(img)
    img = img.sub_(0.5).div_(0.5)
    return img


def normalize_pad(img, max_size, pad_type='right'):
    """Normaliza y añade padding a la imagen.
    Args:
        img (PIL.Image): Imagen de entrada.
        max_size (tuple): Tamaño máximo al que debe ajustarse la imagen.
        pad_type (str): Tipo de padding (actualmente solo soporta 'right').

    Returns:
        torch.Tensor: Imagen con padding añadido y normalizada.
        
    """
    to_tensor = transforms.ToTensor()
    img = to_tensor(img)
    img = img.sub_(0.5).div_(0.5)
    c, h, w = img.size()
    pad_img = torch.FloatTensor(*max_size).fill_(0)
    pad_img[:, :, :w] = img  # Padding hacia la derecha
    if max_size[2] != w:  # Añadir padding de borde
        pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, max_size[2] - w)
    return pad_img


def align_collate(images, imgH, imgW, keep_ratio_with_pad):
    """Ajusta el tamaño de un conjunto de imágenes para ser procesadas en lotes.
    Args:
        images (list): Lista de imágenes.
        imgH (int): Altura de la imagen ajustada.
        imgW (int): Ancho de la imagen ajustada.
        keep_ratio_with_pad (bool): Mantener el aspecto con padding si es True.

    Returns:
        torch.Tensor: Tensor con las imágenes ajustadas y apiladas en un lote.
    """
    if keep_ratio_with_pad:
        max_size = (3 if images[0].mode == 'RGB' else 1, imgH, imgW)
        resized_images = []
        for image in images:
            w, h = image.size
            ratio = w / float(h)
            resized_w = (
                imgW if math.ceil(imgH * ratio) > imgW else math.ceil(imgH * ratio)
            )
            resized_image = image.resize((resized_w, imgH), Image.BICUBIC)
            resized_images.append(normalize_pad(resized_image, max_size))
        image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
    else:
        image_tensors = torch.cat(
            [resize_normalize(image, (imgW, imgH)).unsqueeze(0) for image in images], 0
        )
    return image_tensors


def load_images_from_path(root, opt):
    """Carga las imágenes desde una ruta de directorio.
    Args:
        root (str): Ruta del directorio que contiene las imágenes.
        opt (Namespace): Opciones de configuración, incluyendo formato
          de imagen y dimensiones.

    Returns:
        list: Lista de tuplas con la imagen y su ruta correspondiente.
    """
    image_path_list = []
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            _, ext = os.path.splitext(name)
            ext = ext.lower()
            if ext in ['.jpg', '.jpeg', '.png']:
                image_path_list.append(os.path.join(dirpath, name))

    image_path_list = natsorted(image_path_list)
    images_with_paths = []
    for path in image_path_list:
        try:
            img = Image.open(path).convert('RGB' if opt.rgb else 'L')
        except IOError:
            print(f'Corrupted image for {path}')
            img = Image.new('RGB' if opt.rgb else 'L', (opt.imgW, opt.imgH))
        images_with_paths.append((img, path))

    return images_with_paths


def basic_image_processing(image: Image.Image, coordinates: List[int]) -> Image.Image:
    """Recorta una imagen usando coordenadas proporcionadas y 
    la convierte a escala de grises.

    Args:
        image: Imagen cargada.
        coordinates: Lista de coordenadas para el recorte [xmin, ymin, xmax, ymax].

    Returns:
        Imagen recortada y convertida a escala de grises.
        
    """
    xmin, ymin, xmax, ymax = coordinates
    cropped_image = image.crop((xmin, ymin, xmax, ymax))

    # Convertir la imagen recortada a escala de grises
    gray_image = cropped_image.convert('L')

    return gray_image
