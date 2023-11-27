import torch

unknown_char = '/'
padding_char = '_'

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
    '_': 29, # padding char
    '/': 30, # unknown char
}

charset_dictionary_reversed = {v: k for k, v in charset_dictionary.items()}


def word_to_tensor(word, padding:int = 50, rnn_shift:int = 0):
    length = max(len(word), padding)
    tensor = torch.full((length, ), charset_dictionary[padding_char], dtype=torch.float)

    for i, char in enumerate(word):
        char_pos = min(i + rnn_shift, length - 1)
        tensor[char_pos] = charset_dictionary[char]
    return tensor


def tensor_to_word(tensor):
    word = ''
    tensor = torch.round(tensor).type(torch.long)
    for i in range(tensor.size(0)):
        try:
            word += charset_dictionary_reversed[tensor[i].item()]
        except KeyError:
            word += unknown_char  # unknown char
    return word
