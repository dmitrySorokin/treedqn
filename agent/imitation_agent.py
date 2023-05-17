from .networks import GNNPolicy

import torch
from torch import nn
import torch.nn.functional as F
from utils import pad_tensor


class ImitationAgent:
    def __init__(self, device):
        self.device = device
        self.net = GNNPolicy(device=device, value_head=False).to(device)
        self.opt = torch.optim.Adam(self.net.parameters())
        self.loss = CrossEntropy()
        self.device = device

    def act(self, obs, action_set, deterministic):
        with torch.no_grad():
            preds = self.net(obs)[action_set.astype('int32')]
        action_idx = torch.argmax(preds)
        action = action_set[action_idx.item()]
        return action

    def update(self, obs):
        self.opt.zero_grad()
        pred = self.net(obs)[obs.candidates]
        target = obs.candidate_choices

        loss = self.loss(pred, target, obs.num_candidates)
        loss.backward()
        self.opt.step()
        return loss.detach().cpu().item()

    def validate(self, obs):
        self.opt.zero_grad()
        with torch.no_grad():
            pred = self.net(obs)[obs.candidates]
            target = obs.candidate_choices

            loss = self.loss(pred, target, obs.num_candidates)
        return loss.detach().cpu().item()

    def save(self, path, epoch_id):
        torch.save(self.net.state_dict(), path + f'/checkpoint_{epoch_id}.pkl')

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()


class CrossEntropy(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, _input, target, num_candidates=None):
        if num_candidates is not None:
            _input = pad_tensor(_input, num_candidates)
        return F.cross_entropy(_input, target, reduction=self.reduction)
