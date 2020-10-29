import torch
import torch.nn as nn
import torch.nn.functional as F


class SimTALayer(nn.Module):

    def __init__(self, d_in=1, d_model=256, lamb=1, beta=0):
        super().__init__()
        self.linear = nn.Linear(d_in, d_model)
        self.lamb = nn.Parameter(torch.tensor(lamb, dtype=torch.float))
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float))

    @staticmethod
    def _get_attn_matrix(tau):
        """
        tau: A time interval vector of dimension batch_size * T;
        return: A time-related attention matrix of dimension batch_size * T * T.
        """
        with torch.no_grad():
            d_t = tau.size(1)

            # Repeat tau d_t times along the second dimension, set all elemen-
            # ts on and below the main diagonal to 0, and compute the cumulat-
            # ive sum along the last dimension. Now the element on the i-th r-
            # ow and j-th column of the matrix equals to the duration between
            # tau_i and tau_j.
            tau = torch.unsqueeze(tau, dim=1).repeat(1, d_t, 1)
            t_attn = torch.cumsum(torch.triu(tau, diagonal=1),
                dim=-1).transpose(1, 2)

        return t_attn

    @staticmethod
    def _apply_ninf_mask(attn):
        NINF = -9e8
        d_t = attn.size(-1)
        ninf_mask = torch.triu(torch.ones(d_t, d_t)).bool().logical_not()\
            .transpose(0, 1).to(attn.device)
        attn.masked_fill_(ninf_mask, NINF)

    def forward(self, x, tau):
        """
        x: A feature vector of dimension B * T * d_in;
        tau: A time interval vector of dimension B * T;
        return: B * T * d_model.
        """
        x = F.elu(self.linear(x))
        t_attn = self._get_attn_matrix(tau)
        t_attn = -F.relu(self.lamb) * t_attn + self.beta
        self._apply_ninf_mask(t_attn)

        return torch.bmm(F.softmax(t_attn, dim=-1), x)


class SimTABlock(nn.Module):

    def __init__(self, d_in=1, d_model=256, lamb=1, beta=0, num_layers=4):
        super().__init__()

        for i in range(num_layers):
            if i == 0:
                self.add_module("simta_{}".format(i),
                    SimTALayer(d_in, d_model, lamb, beta))
            else:
                self.add_module("simta_{}".format(i),
                    SimTALayer(d_model, d_model, lamb, beta))

    def forward(self, x, tau):
        for _, module in self.named_children():
            x = module(x, tau)

        return x


class SimTANet(nn.Module):

    def __init__(self, d_in=1, d_model=256, d_out=3,
            lamb=1, beta=0, num_layers=4):
        super().__init__()
        
        self.linear = nn.Linear(d_in, d_model)
        self.simta = SimTABlock(d_model, d_model, lamb, beta, num_layers)
        self.regressor = nn.Linear(d_model, d_out)
    
    def forward(self, x, tau):
        if len(x.size()) == 2:
            x = torch.unsqueeze(x, dim=-1)
        x = self.linear(x)
        # x = self.simta(x, tau).mean(dim=1)
        x = self.simta(x, tau)[:, -1, :]
        y = self.regressor(x)

        return y


if __name__ == "__main__":
    t = torch.tensor([0, 1, 2, 3]).view(1, -1)
    attn = SimTALayer._get_attn_matrix(t)
    SimTALayer._apply_ninf_mask(attn)
    print(attn.numpy())
