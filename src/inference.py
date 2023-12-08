from charset import Charset, TaskBinaryClassification
import helpers
from net_definitions import *


def main():
    model, _ = helpers.load_model(GRUNetBinaryEmbeding, 
        'models/020_gru_binary_emb_256h_29000data_dropout_slower_testing')
        # 'models/019_gru_binary_emb_256h_29000data_dropout/torch_gru_256hid_250batch_500epochs.pt')
    model.eval()
    print('Model:')
    print(model)
    print()

    while True:
        word = input('Enter a word (q to exit): ')
        if not word:
            continue
        if word == 'q':
            break

        out = helpers.transcribe_word(model, word, Charset(TaskBinaryClassification()))
        print(out)

if __name__ == '__main__':
    main()
