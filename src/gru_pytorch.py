"""Simple GRU model for splitting words to syllables.

Inspired by: https://blog.floydhub.com/gru-with-pytorch/ weather forecasting example."""
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from dataset import Dataset


class GRUNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256, output_dim=1, n_layers=1, drop_prob=0.2, device='cpu', batch_size=32):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.batch_size = batch_size

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        print(f'FORWARD! x({x.shape}): {x}')
        out, h = self.gru(x, h)
        print(f'out({out.shape}): {out}')
        print(f'h({h.shape}): {h}')
        out = self.fc(self.relu(out[:,-1]))
        print(f'fc(relu(out)) ({out.shape}): {out}')
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden


def train(train_data: Dataset, learn_rate, hidden_dim=256, EPOCHS=5, device='cpu', batch_size=32):
    # Instantiating the models
    model = GRUNet()
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

    trn_dataset = Dataset(trn_file)
    val_dataset = Dataset(val_file)

    print(f'Loaded {len(trn_dataset)} training and {len(val_dataset)} validation samples.')
    # print('Loaded datasets!!')
    # print(len(trn_dataset))
    # for i in range(10):
    #     print(trn_dataset[i])

    # trn_data = pd.read_csv(trn_file, header=None).rename_columns({0: 'original'})
    # val_data = pd.read_csv(val_file, header=None).rename_columns({0: 'original'})
    # print(f'Loaded {len(trn_data)} training and {len(val_data)} validation samples.')

    batch_size = 32

    # train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    # train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    print('Everything ready!! Net definitions, training and evaluation functions defined.')

    gru_model = train(trn_dataset, learn_rate=0.001, device=device, batch_size=batch_size)
    # gru_outputs, targets, gru_sMAPE = evaluate(gru_model, test_x, test_y, label_scalers)


if __name__ == '__main__':
    main()