"""Tres modelos base de DL: MLP sobre rezagos, LSTM de una capa y CNN 1D poco profunda."""
import torch
import torch.nn as nn
from . import config as C


class MLP(nn.Module):
    def __init__(self, hidden=64, layers=1, dropout=0.0, L=C.L, H=C.H, **_):
        super().__init__()
        mods, d = [], L
        for _i in range(layers):
            mods += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        mods.append(nn.Linear(d, H))
        self.net = nn.Sequential(*mods)

    def forward(self, x):          # x: (B, L)
        return self.net(x)


class LSTM1(nn.Module):
    def __init__(self, hidden=32, dropout=0.0, H=C.H, **_):
        super().__init__()
        self.rnn = nn.LSTM(1, hidden, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, H)

    def forward(self, x):
        out, _ = self.rnn(x.unsqueeze(-1))
        return self.head(self.drop(out[:, -1]))


class CNN1D(nn.Module):
    def __init__(self, channels=32, kernel=3, convs=1, dropout=0.0, L=C.L, H=C.H, **_):
        super().__init__()
        mods, cin = [], 1
        for _i in range(convs):
            mods += [nn.Conv1d(cin, channels, kernel, padding=kernel // 2), nn.ReLU()]
            cin = channels
        self.conv = nn.Sequential(*mods)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(channels * L, H)

    def forward(self, x):
        z = self.conv(x.unsqueeze(1))
        return self.head(self.drop(z.flatten(1)))


MODELS = {"MLP": MLP, "LSTM": LSTM1, "CNN1D": CNN1D}

# Espacios de busqueda comparables en amplitud (cada uno con 5 dimensiones; ver Cap. 5)
SPACES = {
    "MLP":   dict(hidden=[32, 64, 128], layers=[1, 2], dropout=[0.0, 0.1, 0.2], lr=[1e-3, 3e-3], weight_decay=[0.0, 1e-4], batch=[32, 64]),
    "LSTM":  dict(hidden=[16, 32, 64], dropout=[0.0, 0.1, 0.2], lr=[1e-3, 3e-3], weight_decay=[0.0, 1e-4], batch=[32, 64], clip=[0.5, 1.0]),
    "CNN1D": dict(channels=[16, 32, 64], kernel=[3, 5], convs=[1, 2], dropout=[0.0, 0.1, 0.2], lr=[1e-3, 3e-3], weight_decay=[0.0, 1e-4], batch=[32, 64]),
}
# Configuracion minima justificada (piloto de referencia)
DEFAULTS = {
    "MLP":   dict(hidden=64, layers=1, dropout=0.1, lr=1e-3, weight_decay=0.0, batch=32),
    "LSTM":  dict(hidden=32, dropout=0.1, lr=1e-3, weight_decay=0.0, batch=32, clip=1.0),
    "CNN1D": dict(channels=32, kernel=3, convs=1, dropout=0.1, lr=1e-3, weight_decay=0.0, batch=32),
}


def n_params(m):
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))
