import os
import re

import torch

from dataset import Dataset
import charset
import helpers
from gru_pytorch import GRUNet


def transcribe_word(model, word:str):
    # prepare word to input into model
    word_flattened = Dataset.flatten_words([word])[0]
    word_tensor = charset.word_to_tensor(word_flattened)
    word_tensor = helpers.transpose(word_tensor)

    # get output from model
    h = model.init_hidden()
    out, _ = model(word_tensor, h)
    return charset.tensor_to_word(out)


def main():
    model_names = [model for model in os.listdir('models') if model.endswith('.pt')]
    model_name = sorted(model_names, key=lambda x: int(re.match(r'\S+_(\d+)epochs', x).groups(1)[0]))[-1]
    print(model_name)
    model, _ = helpers.load_model(model_name, GRUNet)

    while True:
        word = 'kulaťoučké'
        word = input('Enter a word (exit with q): ')
        if not word:
            continue
        if word == 'q':
            break

        out = transcribe_word(model, word)
        print(out)

if __name__ == '__main__':
    main()
