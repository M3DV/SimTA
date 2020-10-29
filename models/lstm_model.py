import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):

    def __init__(self, d_in=1, d_hidden=256, d_out=3,
            num_layers=4, use_tau=False, use_time_stamp=False):
        super().__init__()

        self.use_tau = use_tau
        self.use_time_stamp = use_time_stamp
        if self.use_tau:
            d_in += 1
        self.lstm = nn.LSTM(d_in, d_hidden, num_layers, batch_first=True)
        self.regressor = nn.Linear(d_hidden * 3, d_out)

    @staticmethod
    def _gen_time_stamp(tau):
        with torch.no_grad():
            return torch.cumsum(tau, dim=1)

    def forward(self, x, tau):
        if self.use_tau:
            output = self._forward_w_tau(x, tau)
        else:
            output = self._forward_wo_tau(x)

        return output

    def _forward_wo_tau(self, x):
        if len(x.size()) == 2:
            x = torch.unsqueeze(x, dim=-1)
        x, _ = self.lstm(x)
        output = self.regressor(x)

        return output

    def _forward_w_tau(self, x, tau):
        if len(x.size()) == 2:
            x = torch.unsqueeze(x, dim=-1)
        if len(tau.size()) == 2:
            tau = torch.unsqueeze(tau, dim=-1)

        if self.use_time_stamp:
            tau = self._gen_time_stamp(tau)

        x = torch.cat([x, tau], dim=-1)
        x, (h, c) = self.lstm(x)
        x = x[:, -1, :]
        h = h[-1, :, :]
        c = c[-1, :, :]
        features = torch.cat([x, h, c], dim=-1)
        output = self.regressor(features)

        return output
