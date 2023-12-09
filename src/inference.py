import re
import pyphen

from charset import Charset, TaskBinaryClassification, TaskNormal
import helpers
from net_definitions import *


class InferenceEngine:
    def __init__(self, gru_old_path: str = 'models/009_torch_gru_8hid_250batch',
                 gru_complex_path: str = 'models/025_gru_binary_emb_256h_29000data_dropout_slower_both'):
        self.gru_old_model = self.load_model(GRUNet, gru_old_path)
        self.gru_complex_model = self.load_model(GRUNetBinaryEmbeding, gru_complex_path)
        self.baseline = pyphen.Pyphen(lang='cs_CZ')

    def load_model(self, cls, path):
        model, _ = helpers.load_model(cls, path)

        if not model:
            raise FileNotFoundError(f'No model found in path: {path}')

        model.eval()
        print(f'Loaded following model from path: {path}')
        print(model)
        print()
        return model

    def __call__(self, word) -> (str, str, str):
        return self.transcribe_word(word)

    def transcribe_word(self, word) -> (str, str, str):
        gru_old = helpers.transcribe_word(self.gru_old_model, word, Charset(TaskNormal))
        gru_complex = helpers.transcribe_word(self.gru_complex_model, word, Charset(TaskBinaryClassification()))
        baseline = self.baseline.inserted(word)
        return gru_old, gru_complex, baseline


def main():
    inference_engine = InferenceEngine()

    while True:
        word = input('Enter a word (q to exit): ')

        if word == 'q':
            break
        if not word or len(word) < 2:
            continue

        gru_old, gru_complex, baseline  = inference_engine.transcribe_word(word)
        if gru_old:
            print(f'old GRU:     {gru_old}')
        if gru_complex:
            print(f'complex GRU: {gru_complex}')
        if baseline:
            print(f'baseline:    {baseline}')

if __name__ == '__main__':
    main()
