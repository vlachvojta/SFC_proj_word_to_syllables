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
    axs[0].plot(trn_losses)
    axs[0].set_title('Trn MSE Loss')
    axs[0].set_xlabel('Epochs')
    axs[1].plot(trn_losses_lev)
    axs[1].set_title('Trn Levenshtein Loss')
    axs[1].set_xlabel('Epochs')
    axs[2].plot([i * view_step for i in range(len(val_losses_lev))], val_losses_lev)
    axs[2].set_title('Val Levenshtein Loss')
    axs[2].set_xlabel('Epochs')
    plt.tight_layout()

    if not os.path.isdir(path):
        os.makedirs(path)
    image_name = f'torch_gru_{hidden_dim}hid_{batch_size}batch_{epoch}epochs_losses.png'
    fig.savefig(os.path.join(path, image_name))


def load_model(model_name, model_class, path:str = 'models'):
    hidden_dim = 0
    match_obj = re.match(r'\S+_(\d+)hid', model_name)
    if match_obj:
        hidden_dim = int(match_obj.groups(1)[0])
    epochs = 0
    match_obj = re.match(r'\S+_(\d+)epochs', model_name)
    if match_obj:
        epochs = int(match_obj.groups(1)[0])

    model = model_class(hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(os.path.join(path, model_name)))
    model.eval()

    print(f'Model loaded from {model_name}')
    print(model)
    print(f'Epochs trained: {epochs}')
    return model, epochs


# def evaluate(model, test_x, test_y, label_scalers):
#     model.eval()
#     outputs = []
#     targets = []
#     start_time = time.time()
#     for i in test_x.keys():
#         inp = torch.from_numpy(np.array(test_x[i]))
#         labs = torch.from_numpy(np.array(test_y[i]))
#         h = model.init_hidden(inp.shape[0])
#         out, h = model(inp.to(device).float(), h)
#         outputs.append(label_scalers[i].inverse_transform(out.cpu().detach().numpy()).reshape(-1))
#         targets.append(label_scalers[i].inverse_transform(labs.numpy()).reshape(-1))
#     print("Evaluation Time: {}".format(str(time.time()-start_time)))
#     sMAPE = 0
#     for i in range(len(outputs)):
#         sMAPE += np.mean(abs(outputs[i]-targets[i])/(targets[i]+outputs[i])/2)/len(outputs)
#     print("sMAPE: {}%".format(sMAPE*100))
#     return outputs, targets, sMAPE