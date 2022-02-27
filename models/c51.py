from abc import ABC
from torch import nn
from torch.nn import functional as F
from .base_model import BaseModel


class C51Model(BaseModel, ABC):
    def __init__(self, state_shape, num_actions, num_atoms, atoms):
        super(C51Model, self).__init__(state_shape)
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.atoms = atoms

        self.logits = nn.Linear(512, num_actions * self.num_atoms)
        nn.init.orthogonal_(self.logits.weight, 0.01)
        self.logits.bias.data.zero_()

    def forward(self, inputs):
        x = inputs / 255
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        logits = self.logits(x)
        return F.softmax(logits.view(-1, self.num_actions, self.num_atoms), dim=-1)  # (Batch size, N_Actions, N_Atoms)

    def get_qvalues(self, x):
        probs = self.forward(x)
        q_values = (self.atoms * probs).sum(dim=-1)  # (Batch size, N_Actions)
        return q_values
