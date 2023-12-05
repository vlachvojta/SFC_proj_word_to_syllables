import os
import re
import unicodedata

import matplotlib.pyplot as plt
import torch
import Levenshtein as lev

import charset
from charset import Task

ERROR_LEV_LOSS = 4242.42

def levenstein_loss(out, label):
    if isinstance(out, list):
        out = ''.join(out)
    if isinstance(label, list):
        label = ''.join(label)

    if len(label) == 0:
        return ERROR_LEV_LOSS
    return 100.0 * lev.distance(out, label) / len(label)


def transpose(data: torch.Tensor):
    """Transpose last dimension of a tensor to the correct shape."""
    if not data.shape[-1] == 1:
        # Transpose: Add a dimension to the end of the tensor
        new_shape = [dim for dim in data.shape] + [1]
        data = torch.reshape(data, new_shape)
        return data

    # De-transpose: Remove the last dimension of the tensor
    new_shape = [dim for dim in data.shape[:-1]]
    data = torch.reshape(data, new_shape)
    return data

def save_model(model, hidden_dim, epoch, batch_size, path:str = '.'):
    model_name = f'torch_gru_{hidden_dim}hid_{batch_size}batch_{epoch}epochs.pt'

    if not os.path.isdir(path):
        os.makedirs(path)

    torch.save(model.state_dict(), os.path.join(path, model_name))
    print(f'Model saved to {model_name}')

def test_val(model, val_data, device, batch_size, task):
    in_words = []
    out_words = []
    labels_words = []

    for i, (x, labels) in enumerate(val_data.batch_iterator(batch_size), start=1):
        out, _ = model(x.to(device))
        inputs = [charset.tensor_to_word(i, task=task) for i in x]
        outputs = [charset.tensor_to_word(o, task=task) for o in out]
        labels = [charset.tensor_to_word(l, task=task) for l in labels]
        in_words += inputs
        out_words += outputs
        labels_words += labels

    in_words = ','.join(in_words)
    out_words = ','.join(out_words)
    labels_words = ','.join(labels_words)

    return levenstein_loss(out_words, labels_words), in_words, out_words, labels_words

def save_out_and_labels(val_out_words, val_labels_words, hidden_dim, epoch, batch_size, path='models'):
    if not os.path.isdir(path):
        os.makedirs(path)

    out_name = f'torch_gru_{hidden_dim}hid_{batch_size}batch_{epoch}epochs_val_out.txt'
    with open(os.path.join(path, out_name), 'w') as f:
        f.write(val_out_words)

    labels_name = f'torch_gru_{hidden_dim}hid_{batch_size}batch_{epoch}epochs_val_labels.txt'
    with open(os.path.join(path, labels_name), 'w') as f:
        f.write(val_labels_words)

    print(f'Out and labels saved to {out_name} and {labels_name}')

def plot_losses(trn_losses, trn_losses_lev, val_losses_lev, hidden_dim, epoch, batch_size,
                view_step: int, path:str = '.'):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    x_ticks = [epoch - len(trn_losses_lev) + e for e in range(len(trn_losses_lev))]
    axs[0].plot(x_ticks, trn_losses)
    axs[0].set_title('Trn MSE Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_yscale('log')
    axs[1].plot(x_ticks, trn_losses_lev)
    axs[1].set_title('Trn Levenshtein Loss')
    axs[1].set_xlabel('Epochs')
    x_ticks = [epoch + (i - len(val_losses_lev)) * view_step for i in range(len(val_losses_lev))]
    axs[1].set_yscale('log')
    axs[2].plot(x_ticks, val_losses_lev)
    axs[2].set_title('Val Levenshtein Loss')
    axs[2].set_xlabel('Epochs')
    axs[2].set_yscale('log')
    plt.tight_layout()

    if not os.path.isdir(path):
        os.makedirs(path)
    image_name = f'torch_gru_{hidden_dim}hid_{batch_size}batch_{epoch}epochs_losses.png'
    fig.savefig(os.path.join(path, image_name))


def load_model(model_class, path:str = 'models'):
    path, model_name = find_last_model(path)
    if not path or not model_name:
        return None, 0

    hidden_dim = 0
    match_obj = re.match(r'\S+_(\d+)hid', model_name)
    if match_obj:
        hidden_dim = int(match_obj.groups(1)[0])
    epochs = 0
    match_obj = re.match(r'\S+_(\d+)epochs', model_name)
    if match_obj:
        epochs = int(match_obj.groups(1)[0])

    model = model_class(hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(os.path.join(path, model_name), map_location=torch.device('cpu')))
    model.eval()

    print(f'Model loaded from {os.path.join(path, model_name)}. {epochs} epochs trained.')
    return model, epochs


def find_last_model(path:str = 'models') -> (str, str):
    if os.path.isdir(path):
        model_names = [model for model in os.listdir(path) if model.endswith('.pt')]
        if model_names:
            last_model = sorted(model_names, key=lambda x: int(re.match(r'\S+_(\d+)epochs', x).groups(1)[0]))[-1]
            return path, last_model
    if os.path.isfile(path) and path.endswith('.pt'):
        return os.path.dirname(path), os.path.basename(path)
    return None, None

def char_classes_to_word(orig: str, classes: list):
    """Converts a list of character classes to a word."""
    word = ''
    for i, c in enumerate(classes):
        if c == '0':
            word += orig[i]
        elif c == '1':
            word += orig[i]
            word += '-'
        else:
            raise ValueError(f'Unknown character class: {c}')
    return word

def transcribe_word(model, word:str, task:Task = Task.NORMAL, tuple_out = False):
    # prepare word to input into model
    word_flattened = flatten_words([word])[0]
    word_tensor = charset.word_to_tensor(word_flattened, task=Task.BINARY_CLASSIFICATION_EMBEDDING)
    # word_tensor = helpers.transpose(word_tensor)

    # get output from model
    with torch.no_grad():
        out, _ = model(word_tensor)
    char_classes = charset.tensor_to_word(out, task=Task.BINARY_CLASSIFICATION_EMBEDDING)

    if tuple_out:
        return char_classes_to_word(word, char_classes), char_classes

    return char_classes_to_word(word, char_classes)

def flatten_words(words):
    original_flat = []
    for word in words:
        flat = unicodedata.normalize('NFD', word).encode('ascii', 'ignore').decode()  # normalize weird czech symbols to basic ASCII symbols
        flat = ''.join(c for c in flat if re.match(r'[a-z\-]', c))
        original_flat.append(flat)
    return original_flat
