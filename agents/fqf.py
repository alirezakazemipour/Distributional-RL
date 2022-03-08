import torch
from models import FQFModel, FractionProposalModel
from common import huber_loss
from .base_agent import BaseAgent


class FQF(BaseAgent):
    def __init__(self, **configs):
        super(FQF, self).__init__(**configs)

        self.online_model = FQFModel(configs["state_shape"],
                                     configs["n_actions"],
                                     configs["n_embedding"],
                                     self.configs["N"]
                                     ).to(self.device)
        self.target_model = FQFModel(configs["state_shape"],
                                     configs["n_actions"],
                                     configs["n_embedding"],
                                     self.configs["N"]
                                     ).to(self.device)
        self.hard_target_update()
        self.optimizer = torch.optim.Adam(self.online_model.parameters(),
                                          self.configs["lr"],
                                          eps=self.configs["adam_eps"]
                                          )
        self.fp_optimizer = torch.optim.RMSprop(self.online_model.fp_layer.parameters(),
                                                self.configs["fp_lr"]
                                                )

    def train(self):
        if len(self.memory) < self.configs["init_mem_size_to_train"]:
            return 0
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = self.unpack_batch(batch)  # noqa

        z, taus, tau_hats = self.online_model(states)
        a = actions[..., None, None].expand(self.batch_size, self.configs["N"], 1).long()
        z = z.gather(dim=-1, index=a)

        with torch.no_grad():
            next_z, *_ = self.target_model(next_states, taus=tau_hats.detach())
            next_qvalues = self.target_model.get_qvalues(next_states)
            next_actions = torch.argmax(next_qvalues, dim=-1)
            next_actions = next_actions[..., None, None].expand(self.batch_size, self.configs["N"], 1)
            next_z = next_z.gather(dim=-1, index=next_actions).squeeze(-1)
            target_z = rewards + self.configs["gamma"] * (~dones) * next_z

        delta = target_z.view(target_z.size(0), 1, target_z.size(-1)) - z
        hloss = huber_loss(delta, self.configs["kappa"])
        rho = torch.abs(tau_hats[..., None].detach() - (delta.detach() < 0).float()) * hloss / self.configs["kappa"]
        loss = rho.sum(1).mean(1).mean()  # sum over N -> mean over N -> mean over batch

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            est_z, *_ = self.online_model(states, use_tau_hats=False)
        a = actions[..., None, None].expand(self.batch_size, self.configs["N"] + 1, 1).long()
        est_z = est_z.gather(dim=-1, index=a)
        fp_grads = 2 * est_z[:, 1:-1] - z[:, 1:].detach() - z[:, :-1].detach()

        taus = taus[:, 1:-1]
        self.fp_optimizer.zero_grad()
        taus.backward(fp_grads.squeeze(-1))
        self.fp_optimizer.step()

        return loss.item()

    @staticmethod
    def get_configs():
        configs = {"n_embedding": 64,
                   "kappa": 1,
                   "N": 32,
                   "lr": 5e-5,
                   "fp_lr": 2.5e-9
                   }
        return configs
