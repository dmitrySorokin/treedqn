from .networks import GNNPolicy

import torch
from torch import nn
import numpy as np
import copy


class FMCTSAgent:
    def __init__(self, device, epsilon):
        super().__init__()
        self.net = GNNPolicy(device=device).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)

        self.target_net_freq = 1000
        self.grad_norm = 50
        self.gamma = 1.0
        self.epsilon = epsilon

    def _predict(self, obs):
        value, adv = self.net(obs)
        return value.mean() + (adv - adv.mean())

    def _act(self, obs, action_set):
        with torch.no_grad():
            preds = self._predict(obs)[action_set.astype('int32')]

        action_idx = torch.argmax(preds)
        action = action_set[action_idx.item()]
        return action

    def act(self, obs, action_set, deterministic):
        if not deterministic and np.random.rand() < self.epsilon:
            return np.random.choice(action_set)

        return self._act(obs, action_set)

    def loss(self, obs, action, ret, treesize):
        qpred = self._predict(obs)[action]
        return ((qpred - ret) ** 2) / treesize


    def update(self, step, batch):
        self.optimizer.zero_grad()

        loss = 0
        
        for trans in zip(batch['obs'], batch['act'], batch['ret'], batch['tree_size']):
            loss += self.loss(*trans)
        
        loss /= len(batch['obs'])
        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_norm)
        self.optimizer.step()

        if step % self.target_net_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        return loss.detach().cpu().item()

    
    def save(self, name):
        torch.save(self.net.state_dict(), name)

    def load(self, path):
        self.net.load_state_dict(
            torch.load(path, map_location=self.net.device)
        )

    def train(self):
        self.net.train()
        self.target_net.train()

    def eval(self):
        self.net.eval()
