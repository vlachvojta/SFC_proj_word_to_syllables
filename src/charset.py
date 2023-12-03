from enum import Enum

import torch

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


class Task(Enum):
    """Enum for training task type."""
    NORMAL = 'normal'
    BINARY_CLASSIFICATION = 'binary_classification'
    BINARY_CLASSIFICATION_EMBEDDING = 'binary_classification_embedding'


def word_to_tensor(word, padding:int = 0, rnn_shift:int = 0, task:Task = Task.NORMAL):
    length = max(len(word), padding)

    if task == Task.NORMAL:
        tensor = torch.full((length, ), charset_dictionary[padding_char], dtype=torch.float)
        for i, char in enumerate(word):
            char_pos = min(i + rnn_shift, length - 1)
            tensor[char_pos] = charset_dictionary[char]
    elif task == Task.BINARY_CLASSIFICATION:
        tensor = torch.full((length, ), 0., dtype=torch.float)
        for i, char in enumerate(word):
            char_pos = min(i + rnn_shift, length - 1)
            tensor[char_pos] = 1. if char == end_of_syllable_char else 0.
    elif task == Task.BINARY_CLASSIFICATION_EMBEDDING:
        tensor = torch.full((length, ), 0, dtype=torch.int)
        for i, char in enumerate(word):
            char_pos = min(i + rnn_shift, length - 1)
            tensor[char_pos] = 1 if char == end_of_syllable_char else 0
    return tensor


def tensor_to_word(tensor, task:Task = Task.NORMAL):
    word = ''
    tensor = torch.round(tensor).type(torch.long)

    if task == Task.NORMAL:
        for i in range(tensor.size(0)):
            try:
                word += charset_dictionary_reversed[tensor[i].item()]
            except KeyError:
                word += unknown_char  # unknown char
    else:
        for i in range(tensor.size(0)):
            try:
                word += str(tensor[i].item())
            except KeyError:
                word += unknown_char  # unknown char
    return word
