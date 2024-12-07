"""
Este script define un modelo OCR (Reconocimiento Óptico de Caracteres) que utiliza
una arquitectura modular compuesta por varias etapas:

Uso:
Este script es parte de un pipeline de OCR y puede ser utilizado en sistemas
de reconocimiento de texto a partir de imágenes, ajustando las configuraciones
del objeto `opt` para adaptarse a diferentes tareas y conjuntos de datos.
"""

# No se puede modificar los nombres de las variables porque son prefijos de los
# pesos del modelo entrenado
# pylint: disable=invalid-name

# pylint: disable=too-many-instance-attributes
# pylint: disable=possibly-used-before-assignment

from torch import nn

from ocr_resnet.model.transformation import TPSSpatialTransformerNetwork
from ocr_resnet.model.feature_extraction import ResNetFeatureExtractor
from ocr_resnet.model.sequence_modeling import BidirectionalLSTM
from ocr_resnet.model.prediction import Attention


class Model(nn.Module):
    """Modelo OCR basado en una arquitectura con etapas de transformación,
    extracción de características, modelado de secuencias y predicción.

    Atributos:
        opt: Objeto que contiene los parámetros de configuración del modelo.
        stages: Diccionario que especifica qué módulos se utilizarán en cada etapa.
        Transformation: Módulo de transformación espacial (opcional).
        FeatureExtraction: Módulo para la extracción de características.
        feature_extraction_output: Salida del extractor de características.
        adaptive_avg_pool: Módulo para aplicar un pooling adaptativo en las
          características extraídas.
        SequenceModeling: Módulo opcional para el modelado de secuencias (BiLSTM).
        sequence_modeling_output: Salida del modelador de secuencias.
        Prediction: Módulo de predicción (Atención).
    """

    def __init__(self, opt):
        """Inicializa el modelo OCR configurando las diferentes
          etapas de procesamiento según las opciones proporcionadas en `opt`.

        Args:
            opt: Objeto que contiene las opciones de configuración
            para las diferentes etapas del modelo
            (Transformation, FeatureExtraction, SequenceModeling, Prediction).
        """
        super().__init__()
        self.opt = opt
        self.stages = {
            'Trans': opt.Transformation,
            'Feat': opt.FeatureExtraction,
            'Seq': opt.SequenceModeling,
            'Pred': opt.Prediction,
        }

        # Transformation
        if opt.Transformation == 'TPS':
            self.Transformation = TPSSpatialTransformerNetwork(
                F=opt.num_fiducial,
                I_size=(opt.imgH, opt.imgW),
                I_r_size=(opt.imgH, opt.imgW),
                I_channel_num=opt.input_channel,
            )
        else:
            print('No Transformation module specified')

        # FeatureExtraction
        if opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNetFeatureExtractor(
                opt.input_channel, opt.output_channel
            )
        else:
            raise Exception('No FeatureExtraction module specified')

        self.feature_extraction_output = opt.output_channel
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(
            (None, 1)
        )  # Transform final (imgH/16-1) -> 1

        # Sequence modeling
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(
                    self.feature_extraction_output, opt.hidden_size, opt.hidden_size
                ),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size),
            )
            self.sequence_modeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.sequence_modeling_output = self.feature_extraction_output

        # Prediction
        if opt.Prediction == 'Attn':
            self.Prediction = Attention(
                self.sequence_modeling_output, opt.hidden_size, opt.num_class
            )
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        """Realiza la inferencia a través de las diferentes etapas del modelo
          (Transformation, FeatureExtraction, SequenceModeling, Prediction).

        Args:
            input: Tensor de entrada con las imágenes del batch.
            text: Texto asociado para la predicción (en modo entrenamiento).
            is_train: Booleano que indica si se está en modo
              de entrenamiento o inferencia.

        Returns:
            prediction: Las predicciones del modelo en la última etapa (Atención).
        """
        # Transformation stage
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        # Feature extraction stage
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.adaptive_avg_pool(
            visual_feature.permute(0, 3, 1, 2)
        )  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        # Sequence modeling stage
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)

        # Prediction stage
        prediction = self.Prediction(
            contextual_feature.contiguous(),
            text,
            is_train,
            batch_max_length=self.opt.batch_max_length,
        )

        return prediction
