"""Modulo para realizar leer placas con OCR"""

# pylint: disable="too-many-locals"
from typing import Dict
import numpy as np

from PIL import Image
import torch
import torch.nn.functional as F

from ocr_resnet.preprocessing import align_collate
from qalpr_v2.metadata.doc_yolo_ocr import KeyOCR


def ocr_placas(
    config_ocr, model, device, converter, cropped_image: Image, placa_id: int
) -> Dict:
    """Realiza OCR sobre la imagen recortada de una placa y devuelve el ID de la placa.

    Args:
        config_ocr: Configuraciones del modelo.
        model: Modelo OCR.
        device: Dispositivo en el que corre el modelo (CPU o GPU).
        converter: Conversor para decodificar la predicción.
        cropped_image: Imagen recortada de la placa.
        placa_id: ID único asociado con la placa.

    Returns:
        output_plates: Diccionario con los resultados del OCR y el ID de la placa.
        
    """
    output_plates = []

    image_tensors = align_collate(
        [cropped_image],
        imgH=config_ocr.imgH,
        imgW=config_ocr.imgW,
        keep_ratio_with_pad=config_ocr.PAD,
    )

    model.eval()
    with torch.no_grad():
        image = image_tensors.to(device)
        length_for_pred = torch.IntTensor([config_ocr.batch_max_length]).to(device)
        text_for_pred = (
            torch.LongTensor(1, config_ocr.batch_max_length + 1).fill_(0).to(device)
        )

        # Pasar la imagen por el modelo OCR
        preds = model(image, text_for_pred, is_train=False)
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        pred = preds_str[0]
        pred_eos = pred.find('[s]')
        pred = pred[:pred_eos]
        pred_max_prob = preds_max_prob[0][:pred_eos]

        if pred_max_prob.size(dim=0):
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]
        else:
            confidence_score = torch.tensor(0, dtype=torch.float)
            pred = ''

        prediction = confidence_score.cpu().numpy()

        output_plates.append(
            {
                KeyOCR.id: placa_id,
                KeyOCR.texto: pred,
                KeyOCR.confianza: np.array_str(prediction),
            }
        )

    return output_plates
