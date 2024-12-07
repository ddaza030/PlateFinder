"""Módulo para implementar un mecanismo de Atención con AttentionCell para modelos
secuencia-a-secuencia.

Este módulo define la clase `Attention` para aplicar mecanismos de atención
y la clase `AttentionCell utilizada dentro de la clase `Attention` para calcular
las puntuaciones de atención y los vectores de contexto.
"""

import torch
from torch import nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Attention(nn.Module):
    """Attention módulo de mecanismos para modelos secuencia a secuencia.

    Args:
        input_size (int): Tamaño de las características de entrada.
        hidden_size (int): Tamaño del estado oculto.
        num_classes (int): Número de clases para la predicción.
    """

    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)

    def _char_to_onehot(self, input_char, onehot_dim=38):
        """Convertir índices de caracteres en vectores de un punto.

        Args:
            input_char (torch.Tensor): Índices de caracteres.
            onehot_dim (int): Dimensión de los vectores unidireccionales.

        Returns:
            torch.Tensor:  Vectores codificados en One-hot.
        """
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(
        self, batch_h, text, is_train=True, batch_max_length=25
    ):  # pylint: disable=too-many-locals
        """Paso hacia delante a través del mecanismo de atención.

        Args:
            batch_h (torch.Tensor): Características contextuales del codificador.
            Forma [batch_size x num_steps x contextual_feature_channels].
            text (torch.Tensor): Índices de texto para cada imagen.
            Forma [tamaño_lote x (longitud_máx+1)].
            is_train (bool): Indicador de si el modelo está en modo de entrenamiento.
            batch_max_length (int): Longitud máxima de las secuencias.

        Returns:
            torch.Tensor: Distribución de probabilidad en cada paso.
            Forma [batch_size x num_steps  x n_class].
        """
        batch_size = batch_h.size(0)
        num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.

        output_hiddens = (
            torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device)
        )
        hidden = (
            torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
            torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
        )

        if is_train:
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(
                    text[:, i], onehot_dim=self.num_classes
                )
                hidden, alpha = self.attention_cell(hidden, batch_h, char_onehots)
                output_hiddens[:, i, :] = hidden[
                    0
                ]  # LSTM hidden index (0: hidden, 1: Cell)
            probs = self.generator(output_hiddens)

        else:
            targets = torch.LongTensor(batch_size).fill_(0).to(device)  # [GO] token
            probs = (
                torch.FloatTensor(batch_size, num_steps, self.num_classes)
                .fill_(0)
                .to(device)
            )

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_h, char_onehots)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x num_classes


class AttentionCell(nn.Module):
    """Attention módulo celular utilizado en el mecanismo de Atención.

    Args:
        input_size (int): Tamaño de las características de entrada.
        hidden_size (int): Tamaño del estado oculto.
        num_embeddings (int): Número de dimensiones de embeddings.
    """

    def __init__(self, input_size, hidden_size, num_embeddings):
        super().__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(
            hidden_size, hidden_size
        )  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_h, char_onehots):
        """Paso hacia delante a través de attention cell.

        Args:
            prev_hidden (tuple): Estado oculto anterior y estado de la célula.
            batch_h (torch.Tensor): Características contextuales del codificador.
            Forma [batch_size x num_encoder_step x num_channel].
            char_onehots (torch.Tensor): Vectores de caracteres codificados en caliente.

        Returns:
            tuple: Actualizados los pesos de estado oculto y atención.
        """
        batch_h_proj = self.i2h(batch_h)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        product = self.score(
            torch.tanh(batch_h_proj + prev_hidden_proj)
        )  # batch_size x num_encoder_step x 1

        alpha = F.softmax(product, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_h).squeeze(
            1
        )  # batch_size x num_channel
        concat_context = torch.cat(
            [context, char_onehots], 1
        )  # batch_size x (num_channel + num_embedding)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha
