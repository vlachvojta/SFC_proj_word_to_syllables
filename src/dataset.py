"""Class defining dataset in this project and its interface."""
import os
import unicodedata
import re

import torch

import charset


class Dataset:
    """Load dataset from file and provide interface for training and evaluating."""
    def __init__(self, path):
        self.path = path

        # read lines of file
        with open(path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

        self.original = [line.strip('\r\n').lower() for line in self.lines]
        self.original_flat = [unicodedata.normalize('NFD', word).encode('ascii', 'ignore').decode()
                              for word in self.original]
        self.original_flat = [''.join(c for c in word if re.match(r'[a-z\-]', c))
                         for word in self.original_flat]

        self.inputs = [word.replace('-', '') for word in self.original_flat]
        self.targets = [re.sub(r'\S\-', '@', word) for word in self.original_flat]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index], self.original[index]

    def batch_iterator(self, batch_size, tensor_output=False):
        """Yield batches as tuple of lists: (trn-in, trn-target)."""
        RNN_SHIFT = 3  # shift for RNN input and target
        batch_size = min(batch_size, len(self))
        batch_count = len(self) // batch_size

        for i in range(batch_count):
            start = batch_size * i
            end = batch_size * (i + 1)
            if not tensor_output:
                yield self.inputs[start:end], self.targets[start:end]
            else:
                max_len = max([len(word) for word in self.inputs[start:end]]) + RNN_SHIFT
                inputs = torch.zeros(batch_size, max_len, 1, dtype=torch.long)
                targets = torch.zeros(batch_size, max_len, 1, dtype=torch.long)
                for j in range(batch_size):
                    inputs[j] = charset.word_to_tensor(self.inputs[start + j], padding=max_len)
                    targets[j+RNN_SHIFT] = charset.word_to_tensor(self.targets[start + j], padding=max_len)
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
