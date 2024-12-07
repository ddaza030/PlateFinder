"""Módulo para la red de transformación espacial TPS (Thin Plate Spline).

Este módulo define una Red de Transformación Espacial basada en TPS utilizada para
rectificar imágenes de entrada. Incluye la clase `TPS_SpatialTransformerNetwork` que
rectifica las imágenes utilizando una transformación TPS, y las clases asociadas
`LocalizationNetwork` y `GridGenerator` que son componentes de la red TPS.

"""

# pylint: disable=C0103
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TPSSpatialTransformerNetwork(nn.Module):
    """Red de rectificación de RARE, es decir, STN basada en TPS.

    Aplica una transformación Thin Plate Spline (TPS) para rectificar las
    imágenes de entrada.

    Args:
        F (int): Número de puntos de control.
        I_size (tuple): (altura, anchura) de la imagen de entrada.
        I_r_size (tuple): (altura, anchura) de la imagen rectificada.
        I_channel_num (int, optional): Número de canales de la imagen de entrada.
        Por defecto es 1.

    Input:
        batch_I (Tensor): Lote de imágenes de entrada con forma
        [batch_size x I_channel_num x I_height x I_width].

    Output:
        batch_I_r (Tensor): Imágenes rectificadas con forma
        [batch_size x I_channel_num x I_r_height x I_r_width].
    """

    def __init__(self, F, I_size, I_r_size, I_channel_num=1):
        super().__init__()
        self.F = F
        self.I_size = I_size
        self.I_r_size = I_r_size
        self.I_channel_num = I_channel_num
        self.LocalizationNetwork = LocalizationNetwork(self.F, self.I_channel_num)
        self.GridGenerator = GridGenerator(self.F, self.I_r_size)

    def forward(self, batch_I):
        """Paso hacia delante a través de la red de transformadores espaciales TPS.

        Args:
            batch_I (Tensor): Lote de imágenes de entrada.

        Returns:
            Tensor: Imágenes rectificadas.
        """
        batch_C_prime = self.LocalizationNetwork(batch_I)
        build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)
        build_P_prime_reshape = build_P_prime.reshape(
            [build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2]
        )

        if torch.__version__ > "1.2.0":
            batch_I_r = F.grid_sample(
                batch_I, build_P_prime_reshape, padding_mode='border', align_corners=True
            )
        else:
            batch_I_r = F.grid_sample(
                batch_I, build_P_prime_reshape, padding_mode='border'
            )

        return batch_I_r


class LocalizationNetwork(nn.Module):
    """Red de localización de RARE, que predice los puntos de control
      para la transformación TPS.

    Args:
        F (int): Número de puntos de control.
        I_channel_num (int): Número de canales de la imagen de entrada.

    Input:
        batch_I (Tensor): Lote de imágenes de entrada con forma
        [batch_size x I_channel_num x I_height x I_width].

    Output:
        batch_C_prime (Tensor): Coordenadas previstas de los puntos fiduciales con forma
        [batch_size x F x 2].
    """

    def __init__(self, F, I_channel_num):
        super().__init__()
        self.F = F
        self.I_channel_num = I_channel_num
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.I_channel_num,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256, self.F * 2)
        self.localization_fc2.weight.data.fill_(0)

        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        self.localization_fc2.bias.data = torch.from_numpy(initial_bias).float().view(-1)

    def forward(self, batch_I):
        """Paso hacia delante a través de la red de localización.

        Args:
            batch_I (Tensor): Lote de imágenes de entrada.

        Returns:
            Tensor: Coordenadas previstas de los puntos de referencia.
        """
        batch_size = batch_I.size(0)
        features = self.conv(batch_I).view(batch_size, -1)
        batch_C_prime = self.localization_fc2(self.localization_fc1(features)).view(
            batch_size, self.F, 2
        )
        return batch_C_prime


class GridGenerator(nn.Module):
    """Generador de cuadrículas de RARE, que produce cuadrículas de muestreo basadas
      en puntos de control previstos.

    Args:
        F (int): Número de puntos de control.
        I_r_size (tuple): Tamaño de la imagen rectificada (alto, ancho).

    Attributes:
        inv_delta_C (Tensor): Matriz delta_C inversa.
        P_hat (Tensor): Matriz P_hat precalculada.

    Methods:
        build_P_prime(batch_C_prime): Generar una cuadrícula de puntos de muestreo
          a partir de los puntos de control previstos.
    """

    def __init__(self, F, I_r_size):
        super().__init__()
        self.eps = 1e-6
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F
        self.C = self._build_C(self.F)
        self.P = self._build_P(self.I_r_width, self.I_r_height)
        self.register_buffer(
            "inv_delta_C", torch.tensor(self._build_inv_delta_C(self.F, self.C)).float()
        )
        self.register_buffer(
            "P_hat", torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float()
        )

    def _build_C(self, F):
        """Construir coordenadas de puntos fiduciales.

        Args:
            F (int): Número de puntos de control.

        Returns:
            numpy.ndarray: Coordenadas de los puntos de referencia.
        """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C

    def _build_inv_delta_C(self, F, C):
        """Construir la matriz inversa delta_C.

        Args:
            F (int): Número de puntos de control.
            C (numpy.ndarray): Coordenadas de los puntos de referencia.

        Returns:
            numpy.ndarray: Matriz delta_C inversa.
        """
        hat_C = np.zeros((F, F), dtype=float)
        for i in range(0, F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C**2) * np.log(hat_C)
        delta_C = np.concatenate(
            [
                np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),
                np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1),
            ],
            axis=0,
        )
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C

    def _build_P(self, I_r_width, I_r_height):
        """Construir la malla de puntos de muestreo.

        Args:
            I_r_width (int): Anchura de la imagen rectificada.
            I_r_height (int): Altura de la imagen rectificada.

        Returns:
            numpy.ndarray: Cuadrícula de puntos de muestreo.
        """
        I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height
        P = np.stack(np.meshgrid(I_r_grid_x, I_r_grid_y), axis=2)
        return P.reshape([-1, 2])

    def _build_P_hat(self, F, C, P):
        """Construir la matriz P_hat.

        Args:
            F (int): Número de puntos de control.
            C (numpy.ndarray): Coordenadas de los puntos de referencia.
            P (numpy.ndarray): Cuadrícula de puntos de muestreo.

        Returns:
            numpy.ndarray: Matriz P_hat.
        """
        n = P.shape[0]
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))
        C_tile = np.expand_dims(C, axis=0)
        P_diff = P_tile - C_tile
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat

    def build_P_prime(self, batch_C_prime):
        """Generar una cuadrícula de puntos de muestreo a partir de los
          puntos de control previstos.

        Args:
            batch_C_prime (Tensor): Puntos de control previstos
              con forma [batch_size x F x 2].

        Returns:
            Tensor: Cuadrícula de puntos de muestreo
              con forma [batch_size x n x 2].
        """
        batch_size = batch_C_prime.size(0)
        batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
        batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)
        batch_C_prime_with_zeros = torch.cat(
            (batch_C_prime, torch.zeros(batch_size, 3, 2).float().to(device)), dim=1
        )
        batch_T = torch.bmm(batch_inv_delta_C, batch_C_prime_with_zeros)
        batch_P_prime = torch.bmm(batch_P_hat, batch_T)

        return batch_P_prime
    