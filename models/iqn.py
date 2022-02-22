import torch
from torch import nn
from torch.nn import functional as F
from abc import ABC
import math


class IQNModel(nn.Module, ABC):
    def __init__(self, state_shape, num_actions, num_embedding):
        super(IQNModel, self).__init__()
        self.num_embedding = num_embedding
        c, w, h = state_shape
        self.conv1 = nn.Conv2d(c, 32, (8, 8), (4, 4))
        self.conv2 = nn.Conv2d(32, 64, (4, 4), (2, 2))
        self.conv3 = nn.Conv2d(64, 64, (3, 3), (1, 1))

        conv1_out_w = self.conv_out_size(w, 8, 4)
        conv1_out_h = self.conv_out_size(h, 8, 4)
        conv2_out_w = self.conv_out_size(conv1_out_w, 4, 2)
        conv2_out_h = self.conv_out_size(conv1_out_h, 4, 2)
        conv3_out_w = self.conv_out_size(conv2_out_w, 3, 1)
        conv3_out_h = self.conv_out_size(conv2_out_h, 3, 1)
        flatten_size = conv3_out_w * conv3_out_h * 64

        self.fc = nn.Linear(flatten_size, 512)
        self.z = nn.Linear(512, num_actions)
        self.phi = nn.Linear(num_embedding, flatten_size)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("relu"))
                layer.bias.data.zero_()

        nn.init.orthogonal_(self.fc.weight, 1.)
        self.fc.bias.data.zero_()
        nn.init.orthogonal_(self.z.weight, 0.01)
        self.z.bias.data.zero_()
        nn.init.orthogonal_(self.phi.weight, gain=nn.init.calculate_gain("relu"))
        self.phi.bias.data.zero_()

    def forward(self, inputs, taus):
        x = inputs / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        state_feats = x.view(x.size(0), -1)

        #  view(...) for broadcasting later when it multiplies to taus
        i_pi = math.pi * torch.arange(self.num_embedding, device=taus.device).view(1, 1, self.num_embedding)
        taus = torch.unsqueeze(taus, -1)
        x = torch.cos(i_pi * taus).view(-1, self.num_embedding)
        phi = F.relu(self.phi(x))

        x = state_feats.view(state_feats.size(0), 1, -1) * phi.view(inputs.size(0), taus.size(1), -1)
        x = x.view(-1, phi.size(-1))
        x = F.relu(self.fc(x))
        z = self.z(x)
        return z.view(inputs.size(0), taus.size(1), -1)

    def get_qvalues(self, x, taus):
        z = self.forward(x, taus)
        q_values = torch.mean(z, dim=1)
        return q_values

    @staticmethod
    def conv_out_size(input_size, kernel_size, stride=1, padding=0):
        return (input_size + 2 * padding - kernel_size) // stride + 1
