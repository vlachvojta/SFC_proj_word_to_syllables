import os
import re

import torch

from dataset import Dataset
import charset
from charset import Task
import helpers
from net_definitions import *


def main():
    model, _ = helpers.load_model(GRUNetBinaryEmbeding, 
        'models/022_gru_binary_emb_256h_29000data_dropout_slower_bigger_test')
        # 'models/019_gru_binary_emb_256h_29000data_dropout/torch_gru_256hid_250batch_500epochs.pt')
        # 'models/020_gru_binary_emb_256h_29000data_dropout_slower')
        # 'models/021_gru_binary_emb_256h_29000data_dropout_slower_bigger')

    model.eval()
    print('Model:')
    print(model)
    print()

    while True:
        word = input('Enter a word (exit with q): ')
        if not word:
            continue
        if word == 'q':
            break

        out = helpers.transcribe_word(model, word)
        print(out)

if __name__ == '__main__':
    main()
