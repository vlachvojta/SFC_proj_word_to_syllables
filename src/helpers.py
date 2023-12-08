import os
import re
import unicodedata

import matplotlib.pyplot as plt
import torch
import Levenshtein as lev

from charset import Charset

ERROR_LEV_LOSS = 4242.42

def levenstein_loss(out, label):
    if isinstance(out, list):
        out = ''.join(out)
    if isinstance(label, list):
        label = ''.join(label)

    if len(label) == 0:
        return ERROR_LEV_LOSS
    return 100.0 * lev.distance(out, label) / len(label)


def get_save_path(training_path: str, hidden_dim, epoch, batch_size, n_layers:int = 1,
                  bidirectional:bool = False, bias: bool = False) -> str:
    directionality = 'bidirectional' if bidirectional else 'unidirectional'
    bias = 'yesbias' if bias else 'nobias'
    model_name = f'torch_gru_{hidden_dim}hid_{n_layers}layers_{directionality}_{bias}_{batch_size}batch_{epoch}epochs'
    return os.path.join(training_path, model_name)

def save_model(model, path:str = 'output_model'):
    export_path = path + '.pt'
    torch.save(model.state_dict(), export_path)
    print(f'Model saved to:\t\t{export_path}')

def save_out_and_labels(val_out_words, val_labels_words, path:str = ''):
    out_path = path + '_val_out.txt'
    with open(out_path, 'w') as f:
        f.write(val_out_words)

    labels_path = path + '_val_labels.txt'
    with open(labels_path, 'w') as f:
        f.write(val_labels_words)

    print(f'Val out saved to:\t{out_path}')
    print(f'Val labels saved to:\t{labels_path}')

def plot_losses(trn_losses, trn_losses_lev, val_losses_lev, epoch,
                view_step: int, path:str = 'output_losses'):
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

    export_path = path + '.png'
    fig.savefig(export_path)


def load_model(model_class, path:str = 'models'):
    path, model_name = find_last_model(path)
    if not path or not model_name:
        return None, 0

    hidden_dim = 8
    match_obj = re.match(r'\S+_(\d+)hid', model_name)
    if match_obj:
        hidden_dim = int(match_obj.groups(1)[0])
    epochs = 0
    match_obj = re.match(r'\S+_(\d+)epochs', model_name)
    if match_obj:
        epochs = int(match_obj.groups(1)[0])
    n_layers = 1
    match_obj = re.match(r'\S+_(\d+)layers', model_name)
    if match_obj:
        n_layers = int(match_obj.groups(1)[0])
    bidirectional = 'bidirectional' in model_name
    bias = 'yesbias' in model_name

    print(f'Loading model from {os.path.join(path, model_name)}')
    print(f'Hidden dim: {hidden_dim}, epochs: {epochs}, n_layers: {n_layers}, bidirectional: {bidirectional}, bias: {bias}')

    model = model_class(hidden_dim=hidden_dim, n_layers=n_layers, bidirectional=bidirectional, bias=bias)
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

def transcribe_word(model, word:str, charset:Charset, device='cpu'):
    # prepare word to input into model
    word_flattened = flatten_words([word])[0]
    word_tensor = charset.word_to_input_tensor(word_flattened).to(device)

    if word_tensor.shape[0] == 0:
        return ''

    with torch.no_grad():
        out, _ = model(word_tensor)

    return charset.tensor_to_word(out, orig_word=word_flattened)

def flatten_words(words):
    original_flat = []
    for word in words:
        flat = unicodedata.normalize('NFD', word).encode('ascii', 'ignore').decode()  # normalize weird czech symbols to basic ASCII symbols
        flat = ''.join(c for c in flat if re.match(r'[a-z\-]', c))
        original_flat.append(flat)
    return original_flat
