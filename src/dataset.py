"""Class defining dataset in this project and its interface."""
import os
import unicodedata
import re

import torch

import charset
import helpers


class Dataset:
    """Load dataset from file and provide interface for training and evaluating."""
    def __init__(self, path, rnn_shift:int = 0, padding:int = 25):
        self.path = path
        self.rnn_shift = rnn_shift  # shift for RNN input and target
        self.padding = padding  # padding for RNN input and target

        # read lines of file
        with open(path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

        self.original = [line.strip('\r\n').lower() for line in self.lines]
        self.original_flat = self.flatten_words()

        self.inputs = []
        self.targets = []
        for word in self.original_flat:
            if self.padding < len(word) + self.rnn_shift:
                word = word[:self.padding - self.rnn_shift]  # Cut word to fit padding
            input_word = word.replace('-', '')
            input_word = charset.padding_char * (self.padding - len(input_word) - self.rnn_shift) + input_word + charset.padding_char * self.rnn_shift
            self.inputs.append(input_word)

            target = re.sub(r'\S\-', '@', word)
            target = charset.padding_char * (self.padding - len(target)) + target
            self.targets.append(target)

    def flatten_words(self):
        original_flat = []
        for word in self.original:
            flat = unicodedata.normalize('NFD', word).encode('ascii', 'ignore').decode()  # normalize weird czech symbols to basic ASCII symbols
            flat = ''.join(c for c in flat if re.match(r'[a-z\-]', c))
            original_flat.append(flat)
        return original_flat    

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index], self.original[index]

    def get_item(self, index):
        x = helpers.transpose(charset.word_to_tensor(self.inputs[index], padding=self.padding))
        target = charset.word_to_tensor(self.targets[index], padding=self.padding, rnn_shift=self.rnn_shift)

        return x, target

    def batch_iterator(self, batch_size, tensor_output=False):
        """Yield batches as tuple of lists: (trn-in, trn-target)."""
        batch_size = min(batch_size, len(self))
        batch_count = len(self) // batch_size

        for i in range(batch_count):
            start = batch_size * i
            end = batch_size * (i + 1)
            if not tensor_output:
                yield self.inputs[start:end], self.targets[start:end]
            else:
                inputs = torch.zeros(batch_size, self.padding, 1, dtype=torch.float)
                targets = torch.zeros(batch_size, self.padding, dtype=torch.float)
                for j in range(batch_size):
                    inputs[j] = helpers.transpose(charset.word_to_tensor(self.inputs[start + j]))
                    targets[j] = charset.word_to_tensor(self.targets[start + j])
                yield inputs, targets


def test():
    dataset = Dataset(os.path.join('dataset', 'ssc_29-06-16', 'set_1000_250', 'val.txt'))
    print(len(dataset))
    for i in range(10):
        print(dataset[i])

    for batch in dataset.batch_iterator(32):
        print(batch)
        break


if __name__ == '__main__':
    test()
