from abc import ABC
from torch import nn
from torch.nn import functional as F
from .base_model import BaseModel


class QRDQNModel(BaseModel, ABC):
    def __init__(self, state_shape, num_actions, num_quantiles):
        super(QRDQNModel, self).__init__(state_shape)
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        self.quantile_probs = 1 / self.num_quantiles

        self.theta = nn.Linear(512, num_actions * self.num_quantiles)
        nn.init.orthogonal_(self.theta.weight, 0.01)
        self.theta.bias.data.zero_()

    def forward(self, inputs):
        x = inputs / 255
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        theta = self.theta(x)
        return theta.view(-1, self.num_quantiles, self.num_actions)  # (Batch size, N_Quantiles, N_Actions)

    def get_qvalues(self, x):
        quantiles = self(x)
        q_values = (self.quantile_probs * quantiles).sum(dim=1)  # (Batch size, N_Actions)
        return q_values
