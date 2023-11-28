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
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_dim)

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
          batch_size=32, save_step=5, view_step=1, training_path:str = None):
    # Instantiating the models
    model, epochs_trained = None, 0
    if training_path and os.path.isdir(training_path):
        if not os.path.isdir(training_path):
            os.makedirs(training_path)
        model, epochs_trained = helpers.load_model(GRUNet, training_path)
        epochs += epochs_trained

    if not model:
        model = GRUNet(hidden_dim=hidden_dim, device=device, batch_size=batch_size)
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
    epoch_outputs = []
    epoch_labels = []
    h = model.init_hidden(batch_size)

    for epoch in range(epochs_trained, epochs + 1):
        start_time = time.time()
        for i, (x, labels) in enumerate(train_data.batch_iterator(batch_size, tensor_output=True), start=1):
            model.zero_grad()
            out, _ = model(x.to(device).float(), h)
            loss = criterion(out, labels.to(device).float())
            loss.backward()

            outputs = [charset.tensor_to_word(o) for o in out]
            epoch_outputs += outputs
            labels = [charset.tensor_to_word(l) for l in labels]
            epoch_labels += labels
            optimizer.step()
        current_time = time.time()
        epoch_times.append(current_time-start_time)
        trn_losses_lev.append(helpers.levenstein_loss(epoch_outputs, epoch_labels))
        trn_losses.append(loss.item())

        if not epoch == epochs_trained and epoch % view_step == 0:
            val_loss_lev, val_out_words, val_labels_words = helpers.test_val(model, val_data, device, batch_size)
            val_losses_lev.append(val_loss_lev)

            print(f"Epoch {epoch}/{epochs}, trn losses: {trn_losses[-1]:.3f}, {trn_losses_lev[-1]:.2f} %, val losses: {val_losses_lev[-1]:.2f} %")
            print(f"Average epoch time in this view_step: {np.mean(epoch_times[-view_step:]):.2f} seconds")
            print('Example:')
            print(f'\tout: {val_out_words[:100]}')
            print(f'\tlab: {val_labels_words[:100]}')
            print('')

        if not epoch == epochs_trained and epoch % save_step == 0:
            helpers.plot_losses(trn_losses, trn_losses_lev, val_losses_lev, hidden_dim, epoch, batch_size, view_step=view_step, path=training_path)
            helpers.save_model(model, hidden_dim, epoch, batch_size, path=training_path)
            helpers.save_out_and_labels(val_out_words, val_labels_words, hidden_dim, epoch, batch_size, path=training_path)

    print(f"Total Training Time: {sum(epoch_times):.2f} seconds. ({np.mean(epoch_times):.2f} seconds per epoch)")

    return model


def main():
    trn_file = os.path.join("dataset", "ssc_29-06-16", "set_1000_250", "trn.txt")
    val_file = os.path.join("dataset", "ssc_29-06-16", "set_1000_250", "val.txt")

    trn_dataset = Dataset(trn_file)
    val_dataset = Dataset(val_file)
    print(f'Loaded {len(trn_dataset)} training and {len(val_dataset)} validation samples.')

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    _ = train(trn_dataset, val_dataset, learn_rate=0.001, device=device, batch_size=64, epochs=1000, save_step=200, view_step=50, training_path='models/005_torch_gru_without_padding')


if __name__ == '__main__':
    main()
