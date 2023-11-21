import torch

def char_to_number(char):
    if ord(char) >= ord('a') and ord(char) <= ord('z'):
        return ord(char) - ord('a')
    if char == '-':
        return 26
    if char == '@':
        return 27
    if char == ' ':  # padding char
        return 28
    return 42  # unknown char

def number_to_char(number):
    if number >= 0 and number <= 25:
        return chr(number + ord('a'))
    if number == 26:
        return '-'
    if number == 27:
        return '@'
    if number == 28:
        return ' '
    if number == 42:
        return '#'
    return '#'

def word_to_tensor(word, padding:int=None):
    if not padding:
        tensor = torch.zeros(len(word), 1, dtype=torch.long)
    else:
        tensor = torch.zeros(padding, 1, dtype=torch.long)
    tensor = torch.fill(tensor, 28)

    for i, char in enumerate(word):
        tensor[i][0] = char_to_number(char)
    return tensor

def tensor_to_word(tensor):
    word = ''
    for i in range(tensor.size(0)):
        word += number_to_char(tensor[i][0].item())
    return word
