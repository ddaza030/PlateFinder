"""Módulo para la red Bidireccional de Memoria Larga a Corto Plazo (LSTM).

Este módulo define la clase `BidirectionalLSTM`, que incluye una capa bidireccional
LSTM bidireccional seguida de una transformación lineal.Procesa secuencias de entrada
para producir características contextuales para tareas de modelado de secuencias.
"""

from torch import nn


class BidirectionalLSTM(nn.Module):
    """Red bidireccional de memoria a largo plazo (LSTM) con una capa lineal.

    Este módulo consta de una LSTM bidireccional seguida de una capa lineal.
    La LSTM procesa las secuencias de entrada y la capa lineal asigna la salida
    al tamaño de salida deseado.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """Paso adelante de la BidirectionalLSTM.

        Args:
            input (Tensor): Tensor de entrada de forma [batch_size x T x input_size].

        Returns:
            Tensor: Tensor de salida de forma [batch_size x T x output_size].
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(
            input
        )  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output
