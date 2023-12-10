import re
import pyphen

from charset import Charset, TaskBinaryClassification, TaskNormal
import helpers
from net_definitions import *


class InferenceEngine:
    def __init__(self, gru_old_path: str = 'models/torch_gru_8hid_250batch_21000epochs.pt',
                 gru_new_path: str = 'models/torch_gru_256hid_2layers_bidirectional_yesbias_250batch_800epochs.pt'):
        self.gru_old_model = self.load_model(GRUNet, gru_old_path)
        self.gru_new_model = self.load_model(GRUNetBinaryEmbeding, gru_new_path)
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
        gru_new = helpers.transcribe_word(self.gru_new_model, word, Charset(TaskBinaryClassification()))
        baseline = self.baseline.inserted(word)
        return gru_old, gru_new, baseline


def main():
    inference_engine = InferenceEngine()

    while True:
        word = input('Enter a word (q to exit): ')

        if word == 'q':
            break
        if not word or len(word) < 2:
            continue

        gru_old, gru_new, baseline  = inference_engine.transcribe_word(word)
        if gru_old:
            print(f'old GRU:  {gru_old}')
        if gru_new:
            print(f'new GRU:  {gru_new}')
        if baseline:
            print(f'baseline: {baseline}')

if __name__ == '__main__':
    main()
