import torch
from torch import nn


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

    def forward(self, x, h=None):
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


class GRUNetBinary(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256, output_dim=1, n_layers=1, device='cpu', batch_size=32):
        super(GRUNetBinary, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.batch_size = batch_size

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.gru = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True, bias=False)
        self.relu2 = nn.ReLU()
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.sigm = nn.Sigmoid()

    def forward(self, x, h=None):
        x = self.encoder(x)
        x = self.relu1(x)
        out, h = self.gru(x, h)
        activated = self.relu2(out)
        out = self.decoder(activated)
        return self.sigm(out), h

    def init_hidden(self, batch_size:int=None):
        weight = next(self.parameters()).data
        if not batch_size:
            return weight.new(self.n_layers, self.hidden_dim).zero_().to(self.device)
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)


class GRUNetBinaryEmbeding(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256, output_dim=1, n_layers=1, device='cpu', batch_size=32):
        super(GRUNetBinaryEmbeding, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.batch_size = batch_size

        self.embed = nn.Embedding(40, hidden_dim)  # 40 is the number of characters in the vocabulary
        self.gru = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True, bias=False)
        self.drop = nn.Dropout(0.2)
        self.relu2 = nn.ReLU()
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.sigm = nn.Sigmoid()

    def forward(self, x, h=None):
        debug = False
        if debug: print('FORWARD!')
        if debug: print(f'x({x.shape}): {x}')
        emb = self.embed(x)
        if debug: print(f'emb({emb.shape}): {emb}')
        out, h = self.gru(emb, h)
        dropped = self.drop(out)
        if debug: print(f'dropped ({dropped.shape}): {dropped}')
        activated = self.relu2(dropped)
        if debug: print(f'activated ({activated.shape}): {activated}')
        out = self.decoder(activated)
        activated = self.sigm(out)
        if debug: print(f'activated ({activated.shape}): {activated}')
        return activated, h

class GRUNetEncDec(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8, output_dim=1, n_layers=1, device='cpu', batch_size=32):
        super(GRUNetEncDec, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.batch_size = batch_size

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True, bias=False)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h):
        x = self.encoder(x)
        out, h = self.gru(x, h)
        activated = self.relu(out)
        out = self.decoder(activated)
        return out, h

    def init_hidden(self, batch_size:int=None):
        weight = next(self.parameters()).data
        if not batch_size:
            return weight.new(self.n_layers, self.hidden_dim).zero_().to(self.device)
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)

