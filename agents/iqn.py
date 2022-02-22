import torch
import numpy as np
import random
from models import IQNModel
from torch import from_numpy
from common import Memory, Transition
from common import huber_loss


class IQN:
    def __init__(self, **configs):
        self.configs = configs
        self.batch_size = configs["batch_size"]
        self.exp_eps = 1
        self.memory = Memory(configs["mem_size"])
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.online_model = IQNModel(configs["state_shape"],
                                     configs["n_actions"],
                                     configs["n_embedding"]
                                     ).to(self.device)
        self.target_model = IQNModel(configs["state_shape"],
                                     configs["n_actions"],
                                     configs["n_embedding"]
                                     ).to(self.device)
        self.hard_target_update()
        self.optimizer = torch.optim.Adam(self.online_model.parameters(),
                                          self.configs["lr"],
                                          eps=self.configs["adam_eps"]
                                          )

    def choose_action(self, state):
        if random.random() < self.exp_eps:
            return random.randint(0, self.configs["n_actions"] - 1)
        else:
            state = np.expand_dims(state, axis=0)
            state = from_numpy(state).byte().to(self.device)
            taus = torch.rand((1, self.configs["K"]), device=self.device)
            with torch.no_grad():
                q_values = self.online_model.get_qvalues(state, taus).cpu()
            return torch.argmax(q_values, -1).item()

    def store(self, state, reward, done, action, next_state):
        assert state.dtype == np.uint8
        assert next_state.dtype == np.uint8
        assert isinstance(done, bool)
        if not isinstance(action, np.uint8):
            action = np.uint8(action)
        ############################
        #  Although we can decrease number of reward's bits but since it turns out to be a numpy array, its
        # overall size increases.
        ############################
        # if not isinstance(reward, np.int8):
        #     reward = np.int8(reward)
        self.memory.add(state, reward, done, action, next_state)

    def unpack_batch(self, batch):
        batch = Transition(*zip(*batch))

        states = from_numpy(np.stack(batch.state)).to(self.device)
        actions = from_numpy(np.stack(batch.action)).to(self.device)
        rewards = from_numpy(np.stack(batch.reward)).view(-1, 1).to(self.device)
        next_states = from_numpy(np.stack(batch.next_state)).to(self.device)
        dones = from_numpy(np.stack(batch.done)).view(-1, 1).to(self.device)
        return states, actions, rewards, next_states, dones

    def hard_target_update(self):
        self.target_model.load_state_dict(self.online_model.state_dict())
        for param in self.target_model.parameters():
            param.requires_grad = False

    def train(self):
        if len(self.memory) < self.configs["init_mem_size_to_train"]:
            return 0
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = self.unpack_batch(batch)

        with torch.no_grad():
            tau_primes = torch.rand((self.batch_size, self.configs["N_prime"]), device=self.device)
            next_z = self.target_model(next_states, tau_primes)
            tau_primes = torch.rand((self.batch_size, self.configs["K"]), device=self.device)
            next_qvalues = self.online_model.get_qvalues(next_states, tau_primes)
            next_actions = torch.argmax(next_qvalues, dim=-1)
            next_actions = next_actions[..., None, None].expand(self.batch_size, self.configs["N_prime"], 1)
            next_z = next_z.gather(dim=-1, index=next_actions).squeeze(-1)
            target_z = rewards + self.configs["gamma"] * (~dones) * next_z

        taus = torch.rand((self.batch_size, self.configs["N"]), device=self.device)
        z = self.online_model(states, taus)
        actions = actions[..., None, None].expand(self.batch_size, self.configs["N"], 1).long()
        z = z.gather(dim=-1, index=actions).squeeze(-1)

        delta = target_z.view(target_z.size(0), 1, target_z.size(-1)) - z.unsqueeze(-1)
        hloss = huber_loss(delta, self.configs["kappa"])
        rho = torch.abs(taus[..., None] - (delta.detach() < 0).float()) * hloss / self.configs["kappa"]
        loss = rho.sum(1).mean(1).mean()  # sum over N -> mean over N_prime -> mean over batch

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @staticmethod
    def get_configs():
        configs = {"adam_eps": 0.01 / 32,
                   "n_embedding": 64,
                   "min_exp_eps": 0.01,
                   "kappa": 1,
                   "N": 32,  # 8 or 32 are acceptable too
                   "N_prime": 64,  # 8 or 32 are acceptable too
                   "K": 32
                   }
        return configs
