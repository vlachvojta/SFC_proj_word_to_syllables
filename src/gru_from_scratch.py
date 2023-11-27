"""Simple GRU model for splitting words to syllables.

Inspired by: https://blog.floydhub.com/gru-with-pytorch/ weather forecasting example."""
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
import helpers

from dataset import Dataset
import charset


class GRUNetFromScratch():
    def __init__(self, input_dim=1, hidden_dim:int = 8, output_dim=1, device='cpu', batch_size=32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        self.batch_size = batch_size

        # TODO create weights for gru layer
        self.gru_x_weights = torch.zeros(3, hidden_dim)  # weights for w_zk, w_rk, w_hk
        self.gru_h_weights = torch.zeros(3)  # weights for v_zk, v_rk, v_hk
        self.fc_weights = torch.zeros(hidden_dim, output_dim)
        torch.nn.init.xavier_uniform_(self.gru_x_weights)
        print('gru_x_weights:')
        print(self.gru_x_weights.shape)
        print(self.gru_x_weights)
        print('gru_h_weights:')
        print(self.gru_h_weights.shape)
        print(self.gru_h_weights)
        print('fc_weights:')
        print(self.fc_weights.shape)
        print(self.fc_weights)

        # self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, dropout=drop_prob)

        # TODO create weights for fc layer
        # self.fc = nn.Linear(hidden_dim, output_dim)

        # TODO maybe some activation function?
        # self.relu = nn.ReLU()

    def __str__(self):
        return (f'GRUNetFromScratch(input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, '
                f'output_dim={self.output_dim}, device={self.device}, batch_size={self.batch_size})')

    def __call__(self, x, h=None):
        if h is None:
            h = self.init_hidden()
        return self.forward(x, h)

    def forward(self, x_batch, h):
        """Get output and hidden state from GRU network."""

        # TODO go though the network and cycle through len of input = count of grus (forward passes)
        out = []
        for word in x_batch:
            out.append([])
            for letter in word:
                # print(f'type of letter: {type(letter.item())}')
                h = self.gru_cell(letter.item(), h)
                # out.append(self.fc(self.relu(h)))
                # TODO out[-1] append linear(activate(current_h))

        # TODO rewrite this for from scratch implementation
        # print(f'FORWARD! x({x.shape}): {x}')
        # out, h = self.gru(x, h)
        # print(f'out({out.shape}): {out}')
        # print(f'h({h.shape}): {h}')
        # out = self.fc(self.relu(out[:,-1]))
        # print(f'fc(relu(out)) ({out.shape}): {out}')
        return out, h

    def gru_cell(self, x, h_prev):
        """GRU cell implementation."""
        x = torch.full((self.input_dim, 1), x, dtype=torch.float)
        # x = torch.full((self.hidden_dim, 1), x, dtype=torch.float)  # TODO This may be wrong, check later
        # self.gru_h_weights for v_zk, v_rk, v_hk
        # self.gru_x_weights for w_zk, w_rk, w_hk

        print('')
        print(f'x ({x.shape}):\n{x}')
        print(f'h_prev ({h_prev.shape}):\n{h_prev}')
        print(f'gru_x_weights ({self.gru_x_weights.shape}):\n{self.gru_x_weights}')

        z_t = torch.sigmoid(torch.mm(self.gru_x_weights[0], x) + torch.mm(self.W_hz, h_prev))  # By ChatGPT...

        u_z_k_t = torch.matmul(self.gru_x_weights[0], x) + self.gru_h_weights[0] * h_prev
        print(f'shape of torch.matmul(self.gru_x_weights[0], x): {torch.matmul(self.gru_x_weights[0], x).shape}')
        asdf = self.gru_h_weights[0] * h_prev
        print(f'shape of self.gru_h_weights[0] * h_prev: {asdf.shape}')
        print(f'u_z_k_t ({u_z_k_t.shape}):\n{u_z_k_t}')
        z_k_t = torch.sigmoid(u_z_k_t)
        print(f'z_k_t ({z_k_t.shape}):\n{z_k_t}')

        u_r_k_t = torch.matmul(self.gru_x_weights[1], x) + self.gru_h_weights[1] * h_prev


        exit(0)
        
        # z_k_t = torch.sigmoid(torch.matmul(self.weights[0], torch.cat((x, h), dim=0)))
        # r_k_t = torch.sigmoid(torch.matmul(self.weights[1], torch.cat((x, h), dim=0)))
        # h_k_t = torch.tanh(torch.matmul(self.weights[2], torch.cat((x, r_k_t * h), dim=0)))
        h = (1 - z_k_t) * h + z_k_t * h_k_t
        return h

    def init_hidden(self, batch_size:int=None):
        if not batch_size:
            hidden = torch.zeros(self.hidden_dim, dtype=torch.float)
        hidden = torch.zeros(batch_size, self.hidden_dim) # torch.zeros(batch_size, self.hidden_dim)  # For batch training
        print(f'init_hidden: {hidden.shape} zeros...')
        return hidden

def train(train_data: Dataset, val_data: Dataset, learn_rate, hidden_dim=8, epochs=5, device='cpu',
          batch_size=1, save_step=5, view_step=1):
    model = GRUNetFromScratch(hidden_dim=hidden_dim, device=device, batch_size=batch_size, output_dim=1)

    print("Starting Training")
    print('')
    epoch_times = []
    trn_losses = []
    trn_losses_lev = []
    val_losses_lev = []
    h = model.init_hidden(batch_size)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for i, (x, labels) in enumerate(train_data.batch_iterator(batch_size, tensor_output=True), start=1):
            print(f'x({x.shape}):')
            print(f'labels({labels.shape})')

            out, _ = model(x, h)
            print(f'out({out.shape}):')
            print(out)
            # Get out loss
            outputs = [charset.tensor_to_word(o) for o in out]
            print(f'outputs({len(outputs)}):')
            print(outputs[:5])
            labels = [charset.tensor_to_word(l) for l in labels]
            print(f'labels({len(labels)}):')
            print(labels[:5])

            # TODO calculate loss using levenstein distance
            trn_losses_lev.append(helpers.levenstein_loss(outputs, labels))
            print(f'Loss: {trn_losses_lev[-1]:.3f} %')

            # TODO backpropagate and update weights
            print('Here you should start updating weights and stuff...')
            exit(0)


            loss = criterion(out, labels.to(device).float())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            trn_losses_lev.append(helpers.levenstein_loss(outputs, labels))
        trn_losses.append(loss.item())
        current_time = time.time()

        if epoch % view_step == 0:
            val_loss_lev, val_out_words, val_labels_words = helpers.test_val(model, val_data, device, batch_size)
            val_losses_lev.append(val_loss_lev)

            print(f"Epoch {epoch}/{epochs}, trn losses: {trn_losses[-1]:.3f}, {trn_losses_lev[-1]:.2f} %, val losses: {val_losses_lev[-1]:.2f} %")
            print(f"Time Elapsed for Epoch: {current_time - start_time:.2f} seconds")
            print('Example:')
            print(f'\tout: {val_out_words[:100]}')
            print(f'\tlab: {val_labels_words[:100]}')
            print('')
        epoch_times.append(current_time-start_time)

        if epoch % save_step == 0:
            helpers.plot_losses(trn_losses, trn_losses_lev, val_losses_lev, hidden_dim, epoch, batch_size, view_step=view_step, path='models')
            helpers.save_model(model, hidden_dim, epoch, batch_size, path='models')
            helpers.save_out_and_labels(val_out_words, val_labels_words, hidden_dim, epoch, batch_size, path='models')

    print(f"Total Training Time: {sum(epoch_times):.2f} seconds. ({sum(epoch_times)/epochs:.2f} seconds per epoch)")

    return model


def main():
    trn_file = os.path.join("dataset", "ssc_29-06-16", "set_1000_250", "trn.txt")
    val_file = os.path.join("dataset", "ssc_29-06-16", "set_1000_250", "val.txt")

    padding = 20

    trn_dataset = Dataset(trn_file, rnn_shift=0, padding=padding)
    val_dataset = Dataset(val_file, rnn_shift=0, padding=padding)
    print(f'Loaded {len(trn_dataset)} training and {len(val_dataset)} validation samples. Such as:')
    
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    gru_model = train(trn_dataset, val_dataset, learn_rate=0.001, device=device, batch_size=1, epochs=500, save_step=50, view_step=5)
    # gru_outputs, targets, gru_sMAPE = evaluate(gru_model, test_x, test_y, label_scalers)


if __name__ == '__main__':
    main()
