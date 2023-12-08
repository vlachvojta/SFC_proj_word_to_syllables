from enum import Enum
from dataclasses import dataclass

import torch
from torch import nn


class tensor_to_word_type(Enum):
    dictionary = 1
    binary = 2


@dataclass
class Task:
    dtype: torch.dtype
    criterion: nn.Module
    transpose_input: bool = True
    transpose_label: bool = True
    label_dtype: torch.dtype = torch.float
    tensor_to_word_type: tensor_to_word_type = tensor_to_word_type.dictionary
    word_to_label_type: tensor_to_word_type = tensor_to_word_type.dictionary


@dataclass
class TaskNormal(Task):
    dtype: torch.dtype = torch.float
    criterion: nn.Module = nn.MSELoss


@dataclass
class TaskBinaryClassification(Task):  # binary classification where each output number represents a class (0 normal, 1 end of syllable)
    dtype: torch.dtype = torch.int
    criterion: nn.Module = nn.BCELoss
    transpose_input: bool = False
    tensor_to_word_type: tensor_to_word_type = tensor_to_word_type.binary
    word_to_label_type: tensor_to_word_type = tensor_to_word_type.binary


class Charset:
    """Class for handling character to tensor and vice versa."""

    unknown_char = '/'
    padding_char = '_'
    end_of_syllable_char = '@'

    charset_dictionary = {
        padding_char: 0, # padding char
        'a': 1,
        'b': 2,
        'c': 3,
        'd': 4,
        'e': 5,
        'f': 6,
        'g': 7,
        'h': 8,
        'i': 9,
        'j': 10,
        'k': 11,
        'l': 12,
        'm': 13,
        'n': 14,
        'o': 15,
        'p': 16,
        'q': 17,
        'r': 18,
        's': 19,
        't': 20,
        'u': 21,
        'v': 22,
        'w': 23,
        'x': 24,
        'y': 25,
        'z': 26,
        end_of_syllable_char: 27,
        unknown_char: 28,
        '-': 29, # reserved for implementation error (this char should not be in the input nor the labels of dataset)
    }

    charset_dictionary_reversed = {v: k for k, v in charset_dictionary.items()}

    def __init__(self, task:Task):
        self.task = task

    def tensor_to_word(self, tensor, orig_word:str = None):
        if tensor.ndim == 2:
            if tensor.shape[-1] > 1:
                raise ValueError(f'Input tensor should be 2D with shape (n, 1) or 1D with shape (n, ), '
                                f'where n is the length of the word, and the values are the classes. Got shape {tensor.shape}.')

            # 2D tensors are transposed to 1D tensors
            if tensor.shape[-1] == 1:
                tensor = Charset.transpose(tensor.clone())

        tensor = torch.round(tensor).type(torch.int)  # round to num representing char or binary class
        out_word = []

        # print(f'self.task.tensor_to_word_type: {self.task.tensor_to_word_type}')

        if self.task.tensor_to_word_type == tensor_to_word_type.dictionary:
            for letter_class in tensor:
                try:
                    output_char = self.charset_dictionary_reversed[letter_class]
                except KeyError:
                    out_word += self.unknown_char
                else:
                    out_word.append(output_char)
        elif self.task.tensor_to_word_type == tensor_to_word_type.binary:
            # print(f'tensor ({tensor.shape}): {tensor}')
            if not orig_word or len(orig_word) != tensor.shape[0]:
                out_word = ''.join([str(c) for c in tensor.tolist()])
                # print(f'DEB: out_word: {out_word}')
            else:
                out_word = ''.join(Charset.char_classes_to_word(orig_word, tensor.tolist()))
        else:
            raise ValueError(f'Unknown tensor_to_word_type: {self.task.tensor_to_word_type}')
        return out_word

    def word_to_tensor_dictionary(self, word, padding:int = 0, rnn_shift:int = 0):
        length = max(len(word), padding)
        tensor = torch.full((length, ), self.charset_dictionary[self.padding_char], dtype=self.task.dtype)

        for i, char in enumerate(word):
            char_pos = min(i + rnn_shift, length - 1)
            tensor[char_pos] = self.charset_dictionary[char]

        return tensor
    
    def word_to_tensor_binary(self, word, padding:int = 0, rnn_shift:int = 0):
        length = max(len(word), padding)

        tensor = torch.full((length, ), 0, dtype=torch.int)
        for i, char in enumerate(word):
            char_pos = min(i + rnn_shift, length - 1)
            tensor[char_pos] = 1 if char == self.end_of_syllable_char else 0

        return tensor

    def word_to_input_tensor(self, word: str, padding:int = 0):
        tensor = self.word_to_tensor_dictionary(word, padding=padding)

        if self.task.transpose_input:
            tensor = self.transpose(tensor)

        return tensor

    def word_to_label_tensor(self, word: str, padding:int = 0):
        if self.task.word_to_label_type == tensor_to_word_type.dictionary:
            label_tensor = self.word_to_tensor_dictionary(word, padding=padding)
        elif self.task.word_to_label_type == tensor_to_word_type.binary:
            label_tensor = self.word_to_tensor_binary(word, padding=padding)
        else:
            raise ValueError(f'Unknown word_to_label_type: {self.task.word_to_label_type}')

        if self.task.transpose_label: 
            label_tensor = self.transpose(label_tensor)
        return label_tensor
    
    def init_input_tensor(self, batch_size:int = None, padding:int = 0):
        init_tensor = torch.zeros(batch_size, padding, dtype=self.task.dtype)

        if self.task.transpose_input:
            init_tensor = self.transpose(init_tensor)

        return init_tensor

    def init_label_tensor(self, batch_size:int = None, padding:int = 0):
        init_tensor = torch.zeros(batch_size, padding, dtype=self.task.label_dtype)
        if self.task.transpose_label:
            init_tensor = self.transpose(init_tensor)
        return init_tensor

    @staticmethod
    def transpose(data: torch.Tensor):
        """Transpose last dimension of a tensor to the correct shape."""
        if not data.shape[-1] == 1:
            # Transpose: Add a dimension to the end of the tensor
            new_shape = [dim for dim in data.shape] + [1]
            data = torch.reshape(data, new_shape)
            return data

        # De-transpose: Remove the last dimension of the tensor
        new_shape = [dim for dim in data.shape[:-1]]
        data = torch.reshape(data, new_shape)
        return data

    @staticmethod
    def char_classes_to_word(orig: str, classes: list):
        """Converts a list of character classes to a word."""
        word = ''
        for i, c in enumerate(classes):
            if c == 0:
                word += orig[i]
            elif c == 1:
                word += orig[i]
                word += '-'
            else:
                raise ValueError(f'Unknown character class: {c}')
        return word
