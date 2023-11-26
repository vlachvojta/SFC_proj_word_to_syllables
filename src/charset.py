import torch

charset_dictionary = {
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
    '-': 27,
    '@': 28, # end of syllable char
    ' ': 29, # padding char
    '#': 30, # unknown char
}

charset_dictionary_reversed = {v: k for k, v in charset_dictionary.items()}


def word_to_tensor(word, padding:int=None):
    if not padding:
        tensor = torch.zeros(len(word), 1, dtype=torch.long)
    else:
        tensor = torch.zeros(padding, 1, dtype=torch.long)
    tensor = torch.fill(tensor, 28)

    for i, char in enumerate(word):
        tensor[i][0] = charset_dictionary[char]
    return tensor


def tensor_to_word(tensor):
    word = ''
    for i in range(tensor.size(0)):
        word += charset_dictionary_reversed(tensor[i][0].item())
    return word
