import torch
from models import C51Model
from .base_agent import BaseAgent


class C51(BaseAgent):
    def __init__(self, **configs):
        super(C51, self).__init__(**configs)
        self.n_atoms = self.configs["n_atoms"]
        self.v_min = self.configs["v_min"]
        self.v_max = self.configs["v_max"]
        self.atoms = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
        self.offset = torch.linspace(0,
                                     (self.batch_size - 1) * self.n_atoms,
                                     self.batch_size
                                     ).long().unsqueeze(1).expand(self.batch_size, self.n_atoms).to(self.device)
        self.online_model = C51Model(configs["state_shape"],
                                     configs["n_actions"],
                                     configs["n_atoms"],
                                     self.atoms
                                     ).to(self.device)
        self.target_model = C51Model(configs["state_shape"],
                                     configs["n_actions"],
                                     configs["n_atoms"],
                                     self.atoms
                                     ).to(self.device)
        self.hard_target_update()
        self.optimizer = torch.optim.Adam(self.online_model.parameters(),
                                          self.configs["lr"],
                                          eps=self.configs["adam_eps"]
                                          )

    def train(self):
        if len(self.memory) < self.configs["init_mem_size_to_train"]:
            return 0
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = self.unpack_batch(batch) # noqa

        with torch.no_grad():
            # next 2 lines can be done more efficiently in a single forward pass!
            next_qvalues = self.target_model.get_qvalues(next_states)
            next_dist = self.target_model(next_states)
            next_actions = torch.argmax(next_qvalues, dim=-1)
            next_actions = next_actions[..., None, None].expand(self.batch_size, 1, self.n_atoms)
            next_chosen_dist = next_dist.gather(dim=1, index=next_actions).squeeze(1)
            target_dist = rewards + self.configs["gamma"] * (~dones) * self.atoms
            target_dist.clamp_(self.v_min, self.v_max)
            b = (target_dist - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            l[(l > 0) * (l == u)] -= 1
            u[(u < (self.atoms - 1)) * (l == u)] += 1

            m = torch.zeros(target_dist.size(), dtype=next_chosen_dist.dtype).to(self.device)
            m.view(-1).index_add_(0,
                                  (l + self.offset).view(-1),
                                  (next_chosen_dist * (u.float() - b)).view(-1)
                                  )
            m.view(-1).index_add_(0,
                                  (u + self.offset).view(-1),
                                  (next_chosen_dist * (b - l.float())).view(-1)
                                  )

        current_dist = self.online_model(states)[range(self.batch_size), actions.long()]
        loss = -(m * torch.log(current_dist + 1e-6)).sum(-1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @staticmethod
    def get_configs():
        configs = {"v_max": 10,
                   "v_min": -10,
                   "n_atoms": 51,
                   "lr": 2.5e-4
                   }
        return configs
