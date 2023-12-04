import os
import re

import torch

from dataset import Dataset
import charset
from charset import Task
import helpers
from net_definitions import *


def read_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.strip('\r\n').replace('-', '') for line in lines]
    return lines


def main():
    model, _ = helpers.load_model(GRUNetBinaryEmbeding, 
        'models/019_gru_binary_emb_256h_29000data_dropout/torch_gru_256hid_250batch_500epochs.pt')
    model.eval()
    print('Model loaded.')
    print(model)
    print()

    words = read_lines('dataset/ssc_29-06-16/ssc_29-06-16_proccessed_shuffled.txt')
    print(f'loaded {len(words)} words such as: {words[:10]}')

    for i, word in enumerate(words[:10]):
        # word += '______'
        out, classes = helpers.transcribe_word(model, word + '______', tuple_out=True)

        if out != word:
            print(f'{word} -> \t{out}')
        
        if i % 1000 == 0:
            print('.', end='', flush=True)

    # for word in words:
    #     out = helpers.transcribe_word(model, word)
    #     print(f'{word} -> \t{out}')
    print()

if __name__ == '__main__':
    main()
