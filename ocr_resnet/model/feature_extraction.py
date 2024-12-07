"""Módulo que contiene la definición de las clases
ResNet_FeatureExtractor, BasicBlock y ResNet para la extraccion de caracteristicas.

*ResNetFeatureExtractor es un extractor de características basado en la arquitectura
ResNet.
*BasicBlock representa un bloque básico de ResNet con conexiones residuales.
*La clase ResNet define el modelo ResNet completo con múltiples capas de BasicBlock.
"""

# pylint: disable=C0103
from torch import nn


class ResNetFeatureExtractor(nn.Module):
    """Extractor de características para FAN (Focusing Attention Network)."""

    def __init__(self, input_channel, output_channel=512):
        """Inicializar el extractor de características ResNet.

        Args:
            input_channel (int):  Número de canales de entrada.
            output_channel (int): Número de canales de salida (por defecto 512).
        """
        super().__init__()
        self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [1, 2, 5, 3])

    def forward(self, input):
        """Paso previo por el extractor de características.

        Args:
            input (Tensor): Tensor de entrada.

        Returns:
            Tensor: Características extraídas.
        """
        return self.ConvNet(input)


class BasicBlock(nn.Module):
    """Bloque básico de ResNet con dos capas convolucionales y
    una conexión de acceso directo."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """Inicializacion de BasicBlock.

        Args:
            inplanes (int): Número de canales de entrada.
            planes (int): Número de canales de salida.
            stride (int): Stride para capas convolucionales (por defecto es 1).
            downsample (nn.Sequential, opcional): Capa de submuestreo.
        """
        super().__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        """3x3 convolución con relleno.

        Args:
            in_planes (int):Número de canales de entrada.
            out_planes (int): Número de canales de salida..
            stride (int): Stride para la convolución (por defecto es 1).

        Returns:
            nn.Conv2d: Capa convolucional.
        """
        return nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

    def forward(self, input_tensor_x):
        """Paso adelante a través del BasicBlock.

        Args:
            x (Tensor): Tensor de entrada.

        Returns:
            Tensor: Tensor de salida tras la convolución y la conexión residual.
        """
        residual = input_tensor_x

        out = self.conv1(input_tensor_x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(input_tensor_x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Arquitectura ResNet con múltiples capas de BasicBlock."""

    def __init__(self, input_channel, output_channel, block, layers):
        """Inicializar el modelo ResNet.

        Args:
            input_channel (int): Número de canales de entrada.
            output_channel (int): Número de canales de salida.
            block (Type[BasicBlock]): Tipo de bloque a utilizar en la red.
            layers (list of int): Lista que especifica el número de bloques de cada capa.
        """
        super().__init__()

        self.output_channel_block = [
            int(output_channel / 4),
            int(output_channel / 2),
            output_channel,
            output_channel,
        ]

        self.inplanes = int(output_channel / 8)
        self.conv0_1 = nn.Conv2d(
            input_channel,
            int(output_channel / 16),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
        self.conv0_2 = nn.Conv2d(
            int(output_channel / 16),
            self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv2d(
            self.output_channel_block[0],
            self.output_channel_block[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(
            block, self.output_channel_block[1], layers[1], stride=1
        )
        self.conv2 = nn.Conv2d(
            self.output_channel_block[1],
            self.output_channel_block[1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(
            block, self.output_channel_block[2], layers[2], stride=1
        )
        self.conv3 = nn.Conv2d(
            self.output_channel_block[2],
            self.output_channel_block[2],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        self.layer4 = self._make_layer(
            block, self.output_channel_block[3], layers[3], stride=1
        )
        self.conv4_1 = nn.Conv2d(
            self.output_channel_block[3],
            self.output_channel_block[3],
            kernel_size=2,
            stride=(2, 1),
            padding=(0, 1),
            bias=False,
        )
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(
            self.output_channel_block[3],
            self.output_channel_block[3],
            kernel_size=2,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        """Create a layer of the network.

        Args:
            block (Type[BasicBlock]): Crear una capa de la red.
            planes (int): Número de canales de salida.
            blocks (int): Número de bloques de la capa.
            stride (int): Stride para el primer bloque de la capa (por defecto es 1).

        Returns:
            nn.Sequential: Contenedor secuencial de bloques.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, input_tensor_x):
        """Paso previo por el modelo ResNet.

        Args:
            x (Tensor): Tensor de entrada.

        Returns:
            Tensor: Tensor de salida tras pasar por la red.
        """
        input_tensor_x = self.conv0_1(input_tensor_x)
        input_tensor_x = self.bn0_1(input_tensor_x)
        input_tensor_x = self.relu(input_tensor_x)
        input_tensor_x = self.conv0_2(input_tensor_x)
        input_tensor_x = self.bn0_2(input_tensor_x)
        input_tensor_x = self.relu(input_tensor_x)

        input_tensor_x = self.maxpool1(input_tensor_x)
        input_tensor_x = self.layer1(input_tensor_x)
        input_tensor_x = self.conv1(input_tensor_x)
        input_tensor_x = self.bn1(input_tensor_x)
        input_tensor_x = self.relu(input_tensor_x)

        input_tensor_x = self.maxpool2(input_tensor_x)
        input_tensor_x = self.layer2(input_tensor_x)
        input_tensor_x = self.conv2(input_tensor_x)
        input_tensor_x = self.bn2(input_tensor_x)
        input_tensor_x = self.relu(input_tensor_x)

        input_tensor_x = self.maxpool3(input_tensor_x)
        input_tensor_x = self.layer3(input_tensor_x)
        input_tensor_x = self.conv3(input_tensor_x)
        input_tensor_x = self.bn3(input_tensor_x)
        input_tensor_x = self.relu(input_tensor_x)

        input_tensor_x = self.layer4(input_tensor_x)
        input_tensor_x = self.conv4_1(input_tensor_x)
        input_tensor_x = self.bn4_1(input_tensor_x)
        input_tensor_x = self.relu(input_tensor_x)
        input_tensor_x = self.conv4_2(input_tensor_x)
        input_tensor_x = self.bn4_2(input_tensor_x)
        input_tensor_x = self.relu(input_tensor_x)

        return input_tensor_x
