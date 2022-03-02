import torch
from models import IQNModel
from common import huber_loss
from .base_agent import BaseAgent


class IQN(BaseAgent):
    def __init__(self, **configs):
        super(IQN, self).__init__(**configs)
        self.online_model = IQNModel(configs["state_shape"],
                                     configs["n_actions"],
                                     configs["n_embedding"],
                                     self.configs["K"]
                                     ).to(self.device)
        self.target_model = IQNModel(configs["state_shape"],
                                     configs["n_actions"],
                                     configs["n_embedding"],
                                     self.configs["K"]
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
        states, actions, rewards, next_states, dones = self.unpack_batch(batch)  # noqa

        with torch.no_grad():
            tau_primes = torch.rand((self.batch_size, self.configs["N_prime"]), device=self.device)
            next_z = self.target_model((next_states, tau_primes))
            next_qvalues = self.target_model.get_qvalues(next_states)
            next_actions = torch.argmax(next_qvalues, dim=-1)
            next_actions = next_actions[..., None, None].expand(self.batch_size, self.configs["N_prime"], 1)
            next_z = next_z.gather(dim=-1, index=next_actions).squeeze(-1)
            target_z = rewards + self.configs["gamma"] * (~dones) * next_z

        taus = torch.rand((self.batch_size, self.configs["N"]), device=self.device)
        z = self.online_model((states, taus))
        actions = actions[..., None, None].expand(self.batch_size, self.configs["N"], 1).long()
        z = z.gather(dim=-1, index=actions)

        delta = target_z.view(target_z.size(0), 1, target_z.size(-1)) - z
        hloss = huber_loss(delta, self.configs["kappa"])
        rho = torch.abs(taus[..., None] - (delta.detach() < 0).float()) * hloss / self.configs["kappa"]
        loss = rho.sum(1).mean(1).mean()  # sum over N -> mean over N_prime -> mean over batch

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @staticmethod
    def get_configs():
        configs = {"n_embedding": 64,
                   "kappa": 1,
                   # N = 64 if earlier phases of training is considered, 32 if final phases is preferable.
                   # according to Fig 2 of the paper.
                   "N": 64,
                   "N_prime": 64,
                   "K": 32,
                   "lr": 5e-5
                   }
        return configs
