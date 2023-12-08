from charset import Charset, TaskBinaryClassification
import helpers
from net_definitions import *


def main():
    model, _ = helpers.load_model(GRUNetBinaryEmbeding, 
        'models/025_gru_binary_emb_256h_29000data_dropout_slower_both')
        # 'models/022_gru_binary_emb_256h_29000data_dropout_slower_bigger_test')
        # 'models/021_gru_binary_emb_256h_29000data_dropout_slower_bigger')
        # 'models/020_gru_binary_emb_256h_29000data_dropout_slower_testing')
        # 'models/020_gru_binary_emb_256h_29000data_dropout_slower')
        # 'models/019_gru_binary_emb_256h_29000data_dropout/torch_gru_256hid_250batch_500epochs.pt')

    if not model:
        raise FileNotFoundError('No model found.')

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
