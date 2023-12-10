from torch import nn


class GRUNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8, output_dim=1, n_layers=1, device='cpu', batch_size=32, bidirectional=False, bias=False):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.batch_size = batch_size

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, bias=False)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h=None):
        out, h = self.gru(x, h)
        activated = self.relu(out)
        out = self.fc(activated)
        return out, h

    def init_hidden(self, batch_size:int=None):
        weight = next(self.parameters()).data
        if not batch_size:
            return weight.new(self.n_layers, self.hidden_dim).zero_().to(self.device)
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)


class GRUNetBinaryEmbeding(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=1, n_layers=1, device='cpu', batch_size=32, bidirectional=False, bias=False):
        # input_dim is max number of characters in the vocabulary

        super(GRUNetBinaryEmbeding, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.batch_size = batch_size

        self.embed = nn.Embedding(input_dim, hidden_dim)  # 40 is the number of characters in the vocabulary
        self.gru = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True, bias=bias, bidirectional=bidirectional)
        self.drop = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
        self.sigm = nn.Sigmoid()

    def forward(self, x, h=None):
        emb = self.embed(x)
        out, h = self.gru(emb, h)
        dropped = self.drop(out)
        activated = self.relu(dropped)
        out = self.decoder(activated)
        bin_class = self.sigm(out)
        return bin_class, h
