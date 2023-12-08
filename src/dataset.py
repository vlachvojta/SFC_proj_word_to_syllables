"""Class defining dataset in this project and its interface."""
import os
import re

from charset import Charset, TaskBinaryClassification
import helpers


class Dataset:
    """Load dataset from file and provide interface for training and evaluating."""
    def __init__(self, path, charset:Charset, rnn_shift:int = 0, padding:int = 0):
        self.path = path
        self.rnn_shift = rnn_shift  # shift for RNN input and target
        self.padding = padding  # padding for RNN input and target
        self.charset = charset
        self.task = charset.task

        # read lines of file
        with open(path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

        self.original = [line.strip('\r\n').lower() for line in self.lines]
        self.original_flat = helpers.flatten_words(self.original)

        self.inputs = []
        self.targets = []
        for word in self.original_flat:
            if self.padding and self.padding < len(word) + self.rnn_shift:
                word = word[:self.padding - self.rnn_shift]  # Cut word to fit padding
            input_word = word.replace('-', '')
            input_word = input_word + charset.padding_char * (self.padding - len(input_word) - self.rnn_shift) + charset.padding_char * self.rnn_shift
            self.inputs.append(input_word)

            target = re.sub(r'\S\-', '@', word)
            target = charset.padding_char * self.rnn_shift + target + charset.padding_char * (self.padding - len(target) - self.rnn_shift)
            self.targets.append(target)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index], self.original[index]

    def get_item_tensors(self, index):
        x = self.charset.word_to_input_tensor(self.inputs[index])
        target = self.charset.word_to_label_tensor(self.targets[index])
        return x, target

    def batch_iterator(self, batch_size):
        """Yield batches as tensors."""
        # Create tensors for input and target, len of tensor is given by max word in batch

        batch_size = min(batch_size, len(self))
        batch_count = len(self) // batch_size

        for i in range(batch_count):
            start = batch_size * i
            end = batch_size * (i + 1)
            max_len = max([len(word) for word in self.inputs[start:end]])

            inputs = self.charset.init_input_tensor(batch_size, max_len)
            targets = self.charset.init_label_tensor(batch_size, max_len)

            for j in range(batch_size):
                inputs[j] = self.charset.word_to_input_tensor(self.inputs[start + j], padding=max_len)
                targets[j] = self.charset.word_to_label_tensor(self.targets[start + j], padding=max_len)

            yield inputs, targets


def test():
    dataset = Dataset(os.path.join('dataset', 'ssc_29-06-16', 'set_1000_250', 'val.txt'), Charset(TaskBinaryClassification))
    print(len(dataset))
    for i in range(10):
        print(dataset[i])

    for batch in dataset.batch_iterator(32):
        print(batch)
        break


if __name__ == '__main__':
    test()
