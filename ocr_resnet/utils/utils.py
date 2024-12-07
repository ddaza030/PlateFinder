"""Este código define una clase llamada `AttnLabelConverter` que es responsable
de convertir entre etiquetas de texto y sus índices correspondientes para un modelo
basado en la atención.

La clase proporciona dos funciones principales:
1. `encode`: Convierte las etiquetas de texto en índices, añadiendo tokens especiales 
   como el token de inicio ([GO]) y el token de fin ([s]).
2. `decode`: Vuelve a convertir los índices en etiquetas de texto originales.
"""

# pylint: disable=C0103
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AttnLabelConverter:
    """Convertir entre texto-etiqueta y texto-índice."""

    def __init__(self, character):
        """Args:
        character (str): Conjunto de los caracteres posibles.
                        [GO] para el token de inicio del decodificador de atención.
                        [s] para el token de fin de frase.
        """
        list_token = ['[GO]', '[s]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {char: i for i, char in enumerate(self.character)}

    def encode(self, text, batch_max_length=25):
        """Convertir texto-etiqueta en texto-índice.

        Args:
            text (list): Etiquetas de texto de cada imagen. [batch_size]
            batch_max_length (int): Longitud máxima de la etiqueta de texto en el lote.

        Returns:
            tuple: Texto como entrada del descodificador de atención
              [batch_size x (max_length + 2)].
            Longitud de la salida también con el token [s]. [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        batch_max_length += 1  # +1 for [GO] token

        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)

        for i, t in enumerate(text):
            text_list = list(t) + ['[s]']
            text_indices = [self.dict[char] for char in text_list]
            batch_text[i][1:1 + len(text_indices)] = torch.LongTensor(text_indices)

        return batch_text.to(device), torch.IntTensor(length).to(device)

    def decode(self, text_index, length):
        """Convertir texto-índice en texto-etiqueta.

        Args:
            text_index (Tensor): Tensor que contiene índices de texto.
            length (Tensor): Longitud de cada secuencia de texto.

        Returns:
            list: Etiquetas de texto descodificadas.
        """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts
