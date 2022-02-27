import torch
from torch import nn
from torch.nn import functional as F
from .base_model import BaseModel
import math


class IQNModel(BaseModel):
    def __init__(self, state_shape, num_actions, num_embedding, k):
        super(IQNModel, self).__init__(state_shape)
        self.num_embedding = num_embedding
        self.k = k

        self.z = nn.Linear(512, num_actions)
        self.phi = nn.Linear(num_embedding, self.flatten_size)

        nn.init.orthogonal_(self.z.weight, 0.01)
        self.z.bias.data.zero_()
        nn.init.orthogonal_(self.phi.weight, gain=nn.init.calculate_gain("relu"))
        self.phi.bias.data.zero_()

    def forward(self, inputs):
        states, taus = inputs
        x = states / 255
        x = self.conv_net(x)
        state_feats = x.view(x.size(0), -1)

        #  view(...) for broadcasting later when it multiplies to taus
        i_pi = math.pi * torch.arange(1, 1 + self.num_embedding, device=taus.device).view(1, 1, self.num_embedding)
        taus = torch.unsqueeze(taus, -1)
        x = torch.cos(i_pi * taus).view(-1, self.num_embedding)
        phi = F.relu(self.phi(x))

        x = state_feats.view(state_feats.size(0), 1, -1) * phi.view(states.size(0), taus.size(1), -1)
        x = x.view(-1, phi.size(-1))
        x = F.relu(self.fc(x))
        z = self.z(x)
        return z.view(states.size(0), taus.size(1), -1)

    def get_qvalues(self, x):
        tau_tildas = torch.rand((x.size(0), self.k), device=x.device)
        z = self.forward((x, tau_tildas))
        q_values = torch.mean(z, dim=1)
        return q_values
