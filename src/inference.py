import re
import pyphen

from charset import Charset, TaskBinaryClassification, TaskNormal
import helpers
from net_definitions import *


class InferenceEngine:
    def __init__(self, gru_complex_path: str = 'models/025_gru_binary_emb_256h_29000data_dropout_slower_both',
                 gru_old_path: str = 'models/002_torch_gru_8hid_32batch-better_but_stilll_shit'):
        self.gru_complex_model = self.load_model(GRUNetBinaryEmbeding, gru_complex_path)
        self.gru_old_model = None  # self.load_model(GRUNet, gru_old_path) # TODO download model from G drive
        self.ref = pyphen.Pyphen(lang='cs_CZ')

    def load_model(self, cls, path):
        model, _ = helpers.load_model(cls, path)

        if not model:
            raise FileNotFoundError(f'No model found in path: {path}')

        model.eval()
        print(f'Loaded following model from path: {path}')
        print(model)
        print()
        return model

    def transcribe_word(self, word) -> (str, str, str):
        gru_complex = helpers.transcribe_word(self.gru_complex_model, word, Charset(TaskBinaryClassification()))
        # gru_old = helpers.transcribe_word(self.gru_old_model, word, Charset(TaskNormal))
        gru_old = ''.join(['_' for c in word])
        ref = self.ref.inserted(word)
        return gru_complex, gru_old, ref


def main():
    inference_engine = InferenceEngine()

    while True:
        word = input('Enter a word (q to exit): ')
        if not word: continue
        if word == 'q': break

        gru_complex, gru_old, ref  = inference_engine.transcribe_word(word)
        if gru_complex:
            print(f'{gru_complex}\t(by complex GRU)')
        if gru_old:
            print(f'{gru_old}\t(by old GRU)')
        if ref:
            print(f'{ref}\t(ref)')

if __name__ == '__main__':
    main()
