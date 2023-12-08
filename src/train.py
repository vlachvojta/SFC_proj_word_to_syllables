"""Simple GRU model for splitting words to syllables.

Inspired by: https://blog.floydhub.com/gru-with-pytorch/ weather forecasting example."""
import os
import time

import numpy as np

import torch
from torch import nn

from dataset import Dataset
from charset import Task, Charset, TaskBinaryClassification
import helpers
from net_definitions import *


def train(net, train_data: Dataset, val_data: Dataset, charset: Charset,
          learn_rate, device='cpu', batch_size=32, epochs=5, save_step=5, view_step=1,
          hidden_dim=8, GRU_layers=1, bidirectional=False, bias=False,
          training_path:str = 'models'):

    model, epochs_trained = None, 0
    if training_path:
        if not os.path.isdir(training_path):
            os.makedirs(training_path)
        else:
            model, epochs_trained = helpers.load_model(net, training_path)
            epochs += epochs_trained

    if not model:
        model = net(hidden_dim=hidden_dim, device=device, batch_size=batch_size, n_layers=GRU_layers, bidirectional=bidirectional, bias=bias)

    model.to(device)
    print(f'Using device: {device}')
    print(f'Using model:\n{model}')

    # Defining loss function and optimizer
    criterion = charset.task.criterion()
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
        export_path = helpers.get_save_path(training_path, hidden_dim, epoch, batch_size, n_layers=GRU_layers, bidirectional=bidirectional, bias=bias)
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

            outputs = [charset.tensor_to_word(o) for o in out]
            epoch_outputs += outputs
            labels = [charset.tensor_to_word(l) for l in labels]
            epoch_labels += labels
        trn_losses_lev.append(helpers.levenstein_loss(epoch_outputs, epoch_labels))
        trn_losses.append(loss.item())

        if epoch % view_step == 0:
            model.eval()
            val_loss_lev, val_in_words, val_out_words, val_labels_words = test_val(model, val_data, device, batch_size, charset)
            val_losses_lev.append(val_loss_lev)

            print(f"Epoch {epoch}/{epochs}, trn losses: {trn_losses[-1]:.5f}, {trn_losses_lev[-1]:.5f} %, val losses: {val_losses_lev[-1]:.3f} %")
            print(f"Average epoch time in this view_step: {np.mean(epoch_times[-view_step:]):.2f} seconds")
            print('Example:')
            print(f'\tin:  {val_in_words[:100]}')
            print(f'\tout: {val_out_words[:100]}')
            print(f'\tlab: {val_labels_words[:100]}')
            print('')

        if epoch % save_step == 0:
            helpers.plot_losses(trn_losses, trn_losses_lev, val_losses_lev, epoch, view_step=view_step, path=export_path)
            helpers.save_model(model, path=export_path)
            helpers.save_out_and_labels(val_out_words, val_labels_words, path=export_path)
        current_time = time.time()
        print(f'epoch time: {current_time-start_time:.2f} seconds')
        epoch_times.append(current_time-start_time)

    print(f"Total Training Time: {sum(epoch_times):.2f} seconds. ({np.mean(epoch_times):.2f} seconds per epoch)")

    return model


def test_val(model, val_data, device, batch_size, charset:Charset):
    in_words = []
    out_words = []
    labels_words = []

    for i, (x, labels) in enumerate(val_data.batch_iterator(batch_size), start=1):
        out, _ = model(x.to(device))
        inputs = [charset.tensor_to_word(i) for i in x]
        outputs = [charset.tensor_to_word(o) for o in out]
        labels = [charset.tensor_to_word(l) for l in labels]

        in_words += inputs
        out_words += outputs
        labels_words += labels

    in_words = ','.join(in_words)
    out_words = ','.join(out_words)
    labels_words = ','.join(labels_words)

    return helpers.levenstein_loss(out_words, labels_words), in_words, out_words, labels_words


def main():
    # TODO move main to the top
    set_size = 'set_30000_8000'
    charset = Charset(TaskBinaryClassification)

    trn_file = os.path.join("dataset", "ssc_29-06-16", set_size, "trn.txt")
    val_file = os.path.join("dataset", "ssc_29-06-16", set_size, "val.txt")

    trn_dataset = Dataset(trn_file, charset=charset)
    val_dataset = Dataset(val_file, charset=charset)
    print(f'Loaded {len(trn_dataset)} training and {len(val_dataset)} validation samples.')

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    train(GRUNetBinaryEmbeding, trn_dataset, val_dataset, charset=charset,
          learn_rate=0.0001, device=device, batch_size=250, epochs=1_000, save_step=50, view_step=10,
          hidden_dim=256, GRU_layers=2, bidirectional=True, bias=True,
          training_path='models/042_gru_binary_emb_256h_29000data_dropout_slower_bigger_TEST')


if __name__ == '__main__':
    main()
