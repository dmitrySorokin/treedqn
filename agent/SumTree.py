import ecole 
import numpy as np


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)

        self.obs = np.zeros(capacity, dtype=ecole.core.observation.NodeBipartiteObs)
        self.rew = np.zeros(capacity, dtype=float)
        self.act = np.zeros(capacity, int)

        self.nextobs = np.zeros(capacity, dtype=list)
        self.nextactset = np.zeros(capacity, dtype=list)
        self.done = np.zeros(capacity, dtype=float)

        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, sample):
        obs, nextobs, nextactset, rew, act, done = sample

        idx = self.write + self.capacity - 1

        self.obs[self.write] = obs
        self.nextobs[self.write] = nextobs
        self.nextactset[self.write] = nextactset
        self.rew[self.write] = rew
        self.act[self.write] = act
        self.done[self.write] = done
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        sample = {'obs': self.obs[dataIdx], 
            'act': self.act[dataIdx], 
            'rew': self.rew[dataIdx],
            'next_obs': self.nextobs[dataIdx], 
            'next_actset': self.nextactset[dataIdx],
            'done': self.done[dataIdx]
        }

        return (idx, self.tree[idx], sample)
