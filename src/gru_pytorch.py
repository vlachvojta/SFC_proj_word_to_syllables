"""Simple GRU model for splitting words to syllables.

Inspired by: https://blog.floydhub.com/gru-with-pytorch/ weather forecasting example."""
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn

from dataset import Dataset
import charset
import helpers


class GRUNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8, output_dim=1, n_layers=1, device='cpu', batch_size=32):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.batch_size = batch_size

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, bias=False)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        # print(f'FORWARD! x({x.shape})') #: {x}')
        out, h = self.gru(x, h)
        # print(f'out({out.shape})')#: {out}')
        # print(f'h({h.shape})')#: {h}')
        activated = self.relu(out)
        # print(f'activated({activated.shape})')#: {activated}')
        out = self.fc(activated)
        # print(f'fc(relu(out)) ({out.shape})')#: {out}')
        return out, h

    def init_hidden(self, batch_size:int=None):
        weight = next(self.parameters()).data
        if not batch_size:
            return weight.new(self.n_layers, self.hidden_dim).zero_().to(self.device)
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)


def train(train_data: Dataset, val_data: Dataset, learn_rate, hidden_dim=8, epochs=5, device='cpu',
          batch_size=32, save_step=5, view_step=1):
    # Instantiating the models
    model = GRUNet(hidden_dim=hidden_dim, device=device, batch_size=batch_size, output_dim=1)
    model.to(device)

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    print("Starting Training")
    print('')
    epoch_times = []
    trn_losses = []
    trn_losses_lev = []
    val_losses_lev = []

    for epoch in range(1,epochs+1):
        start_time = time.time()
        h = model.init_hidden(batch_size)
        for i, (x, labels) in enumerate(train_data.batch_iterator(batch_size, tensor_output=True), start=1):
            # print(f'x({x.shape})')
            # print(f'labels({labels.shape})')
            h = h.data
            model.zero_grad()

            out, h = model(x.to(device).float(), h)

            # print('Trying to get loss:')
            # print(f'out({out.shape})')
            # print(f'labels({labels.shape})')

            loss = criterion(out, labels.to(device).float())
            # print(f'loss: {loss}')
            loss.backward()

            outputs = [charset.tensor_to_word(o) for o in out]
            labels = [charset.tensor_to_word(l) for l in labels]
            # print(f'Levenshtein loss: {trn_losses[-1]:.3f} %')

            optimizer.step()
        trn_losses_lev.append(helpers.levenstein_loss(outputs, labels))
        trn_losses.append(loss.item())
        current_time = time.time()

        if epoch % view_step == 0:
            val_loss_lev, val_out_words, val_labels_words = test_val(model, val_data, device, batch_size)
            val_losses_lev.append(val_loss_lev)

            print(f"Epoch {epoch}/{epochs}, trn losses: {trn_losses[-1]:.3f}, {trn_losses_lev[-1]:.2f} %, val losses: {val_losses_lev[-1]:.2f} %")
            print(f"Time Elapsed for Epoch: {current_time - start_time:.2f} seconds")
            print('Example:')
            print(f'\tout: {val_out_words[:100]}')
            print(f'\tlab: {val_labels_words[:100]}')
            print('')
        epoch_times.append(current_time-start_time)

        if epoch % save_step == 0:
            plot_losses(trn_losses, trn_losses_lev, val_losses_lev, hidden_dim, epoch, batch_size, view_step=view_step, path='models')
            save_model(model, hidden_dim, epoch, batch_size, path='models')
            save_out_and_labels(val_out_words, val_labels_words, hidden_dim, epoch, batch_size, path='models')

    print(f"Total Training Time: {sum(epoch_times):.2f} seconds. ({sum(epoch_times)/epochs:.2f} seconds per epoch)")

    return model

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

    return helpers.levenstein_loss(out_words, labels_words), out_words, labels_words

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


def main():
    trn_file = os.path.join("dataset", "ssc_29-06-16", "set_1000_250", "trn.txt")
    val_file = os.path.join("dataset", "ssc_29-06-16", "set_1000_250", "val.txt")

    padding = 20

    trn_dataset = Dataset(trn_file, rnn_shift=0, padding=padding)
    val_dataset = Dataset(val_file, rnn_shift=0, padding=padding)
    print(f'Loaded {len(trn_dataset)} training and {len(val_dataset)} validation samples.')

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # gru_model = train(trn_dataset, val_dataset, learn_rate=0.001, device=device, batch_size=32, epochs=500, save_step=50, view_step=5, padding=padding)
    gru_model = train(trn_dataset, val_dataset, learn_rate=0.001, device=device, batch_size=32, epochs=2000, save_step=200, view_step=40)
    # gru_outputs, targets, gru_sMAPE = evaluate(gru_model, test_x, test_y, label_scalers)


if __name__ == '__main__':
    main()
