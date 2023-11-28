import os
import re

import matplotlib.pyplot as plt
import torch
import Levenshtein as lev

import charset


def levenstein_loss(a, b):
    if isinstance(a, list):
        a = ''.join(a)
    if isinstance(b, list):
        b = ''.join(b)
    return 100.0 * lev.distance(a, b) / len(b)


def transpose(data: torch.Tensor):
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

def test_val(model, val_data, device, batch_size):
    out_words = []
    labels_words = []

    for i, (x, labels) in enumerate(val_data.batch_iterator(batch_size, tensor_output=True), start=1):
        h_start = model.init_hidden(batch_size=batch_size)
        out, _ = model(x.to(device).float(), h_start.to(device).float())
        outputs = [charset.tensor_to_word(o) for o in out]
        labels = [charset.tensor_to_word(l) for l in labels]
        out_words += outputs
        labels_words += labels

    out_words = ','.join(out_words)
    labels_words = ','.join(labels_words)

    return levenstein_loss(out_words, labels_words), out_words, labels_words

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
    axs[1].plot(x_ticks, trn_losses_lev)
    axs[1].set_title('Trn Levenshtein Loss')
    axs[1].set_xlabel('Epochs')
    x_ticks = [epoch + (i - len(val_losses_lev)) * view_step for i in range(len(val_losses_lev))]
    axs[2].plot(x_ticks, val_losses_lev)
    axs[2].set_title('Val Levenshtein Loss')
    axs[2].set_xlabel('Epochs')
    plt.tight_layout()

    if not os.path.isdir(path):
        os.makedirs(path)
    image_name = f'torch_gru_{hidden_dim}hid_{batch_size}batch_{epoch}epochs_losses.png'
    fig.savefig(os.path.join(path, image_name))


def load_model(model_class, path:str = 'models'):
    model_name = find_last_model(path)
    if not model_name:
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

    print(f'Model loaded from {model_name}')
    print(model)
    print(f'Epochs trained: {epochs}')
    return model, epochs


def find_last_model(path:str = 'models'):
    model_names = [model for model in os.listdir(path) if model.endswith('.pt')]
    if not model_names:
        return None
    return sorted(model_names, key=lambda x: int(re.match(r'\S+_(\d+)epochs', x).groups(1)[0]))[-1]
