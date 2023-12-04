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
from charset import Task
import helpers


from net_definitions import *


def train(net, train_data: Dataset, val_data: Dataset, task: Task, learn_rate, hidden_dim=8, epochs=5, device='cpu',
          batch_size=32, save_step=5, view_step=1, training_path:str = None):
    # Instantiating the models
    model, epochs_trained = None, 0
    if training_path and os.path.isdir(training_path):
        if not os.path.isdir(training_path):
            os.makedirs(training_path)
        model, epochs_trained = helpers.load_model(net, training_path)
        epochs += epochs_trained

    if not model:
        model = net(hidden_dim=hidden_dim, device=device, batch_size=batch_size)

    model.to(device)
    print(f'Using device: {device}')
    print(f'Using model:\n{model}')

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    if task == Task.BINARY_CLASSIFICATION or task == Task.BINARY_CLASSIFICATION_EMBEDDING:
        criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    print("Starting Training")
    print('')
    epoch_times = []
    trn_losses = []
    trn_losses_lev = []
    val_losses_lev = []
    h = None  # model.init_hidden(batch_size).to(device)

    for epoch in range(epochs_trained, epochs + 1):
        epoch_outputs = []
        epoch_labels = []
        start_time = time.time()
        model.train()

        for i, (x, labels) in enumerate(train_data.batch_iterator(batch_size), start=1):
            model.zero_grad()
            out, _ = model(x.to(device), h)
            loss = criterion(out, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            outputs = [charset.tensor_to_word(o, task=task) for o in out]
            epoch_outputs += outputs
            labels = [charset.tensor_to_word(l, task=task) for l in labels]
            epoch_labels += labels
        trn_losses_lev.append(helpers.levenstein_loss(epoch_outputs, epoch_labels))
        trn_losses.append(loss.item())

        if epoch % view_step == 0:
            model.eval()
            val_loss_lev, val_in_words, val_out_words, val_labels_words = helpers.test_val(model, val_data, device, batch_size, task=task)
            val_losses_lev.append(val_loss_lev)

            print(f"Epoch {epoch}/{epochs}, trn losses: {trn_losses[-1]:.3f}, {trn_losses_lev[-1]:.2f} %, val losses: {val_losses_lev[-1]:.2f} %")
            print(f"Average epoch time in this view_step: {np.mean(epoch_times[-view_step:]):.2f} seconds")
            print('Example:')
            print(f'\tin:  {val_in_words[:100]}')
            print(f'\tout: {val_out_words[:100]}')
            print(f'\tlab: {val_labels_words[:100]}')
            print('')

        if epoch % save_step == 0:
            helpers.plot_losses(trn_losses, trn_losses_lev, val_losses_lev, hidden_dim, epoch, batch_size, view_step=view_step, path=training_path)
            helpers.save_model(model, hidden_dim, epoch, batch_size, path=training_path)
            helpers.save_out_and_labels(val_out_words, val_labels_words, hidden_dim, epoch, batch_size, path=training_path)
        current_time = time.time()
        print(f'epoch time: {current_time-start_time:.2f} seconds')
        epoch_times.append(current_time-start_time)

    print(f"Total Training Time: {sum(epoch_times):.2f} seconds. ({np.mean(epoch_times):.2f} seconds per epoch)")

    return model


def main():
    set_size = 'set_30000_8000'
    trn_file = os.path.join("dataset", "ssc_29-06-16", set_size, "trn.txt")
    val_file = os.path.join("dataset", "ssc_29-06-16", set_size, "val.txt")

    task = Task.BINARY_CLASSIFICATION_EMBEDDING
    trn_dataset = Dataset(trn_file, task=task)
    val_dataset = Dataset(val_file, task=task)
    print(f'Loaded {len(trn_dataset)} training and {len(val_dataset)} validation samples.')

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    _ = train(GRUNetBinaryEmbeding, trn_dataset, val_dataset, task, learn_rate=0.0001, hidden_dim=256, device=device,
              batch_size=250, epochs=2_000, save_step=50, view_step=10,
              training_path='models/020_gru_binary_emb_256h_29000data_dropout_slower')


if __name__ == '__main__':
    main()
