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


class GRUNetFromScratch(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256, output_dim=1, device='cpu', batch_size=32):
        self.hidden_dim = hidden_dim
        self.device = device
        self.batch_size = batch_size

        self.weights = torch.zeros(4, hidden_dim)
        torch.nn.init.xavier_uniform_(self.weights)

        # TODO create weights for gru layer
        # self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, dropout=drop_prob)

        # TODO create weights for fc layer
        # self.fc = nn.Linear(hidden_dim, output_dim)

        # TODO maybe some activation function?
        # self.relu = nn.ReLU()

    def __str__(self):
        return (f'GRUNetFromScratch(input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, '
                f'output_dim={self.output_dim}, drop_prob={self.drop_prob}, device={self.device}, batch_size={self.batch_size})')

    def forward(self, x, h):
        # TODO rewrite this for from scratch implementation
        print(f'FORWARD! x({x.shape}): {x}')
        out, h = self.gru(x, h)
        print(f'out({out.shape}): {out}')
        print(f'h({h.shape}): {h}')
        out = self.fc(self.relu(out[:,-1]))
        print(f'fc(relu(out)) ({out.shape}): {out}')
        return out, h


def train(train_data: Dataset, learn_rate, hidden_dim=256, EPOCHS=5, device='cpu', batch_size=32):
    # Instantiating the models
    model = GRUNetFromScratch()
    model.to(device)

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    print("Starting Training")
    epoch_times = []
    # Start training loop
    for epoch in range(1,EPOCHS+1):
        start_time = time.time()
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        counter = 0
        for x, label in train_data.batch_iterator(batch_size, tensor_output=True):
            print(f'x({x.shape}):')
            print(f'label({label.shape})')
            counter += 1
            h = h.data
            model.zero_grad()
            
            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if counter%200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), avg_loss/counter))
        current_time = time.time()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss/len(train_loader)))
        print("Time Elapsed for Epoch: {} seconds".format(str(current_time-start_time)))
        epoch_times.append(current_time-start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    return model


def evaluate(model):
    pass


def main():
    trn_file = os.path.join("dataset", "ssc_29-06-16", "set_1000_250", "trn.txt")
    val_file = os.path.join("dataset", "ssc_29-06-16", "set_1000_250", "val.txt")

    trn_dataset = Dataset(trn_file)
    val_dataset = Dataset(val_file)

    print(f'Loaded {len(trn_dataset)} training and {len(val_dataset)} validation samples.')
    for i in range(10):
        print(trn_dataset[i])

    batch_size = 32
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    for batch in trn_dataset.batch_iterator(batch_size, tensor_output=True):
        print(f'batch inputs: ({batch[0].shape})')
        print(f'batch targets: ({batch[1].shape})')
        print(batch[0])
        print(batch[1])
        break

    gru_model = train(trn_dataset, learn_rate=0.001, device=device, batch_size=batch_size)
    # gru_outputs, targets, gru_sMAPE = evaluate(gru_model, test_x, test_y, label_scalers)


if __name__ == '__main__':
    main()
