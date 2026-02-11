import numpy as np
import ecole
from agent.SumTree import SumTree


class ReplayBuffer:
    def __init__(self, max_size=50000, start_size=10, batch_size=32):
        self.max_size = max_size
        self.start_size = start_size
        self.size = 0
        self.insert_idx = 0
        self.batch_size = batch_size
        self.obs = np.zeros(max_size, dtype=ecole.core.observation.NodeBipartiteObs)
        self.rew = np.zeros(max_size, dtype=float)
        self.act = np.zeros(max_size, int)

        self.nextobs = np.zeros(max_size, dtype=list)
        self.nextactset = np.zeros(max_size, dtype=list)
        self.done = np.zeros(max_size, dtype=float)

    def add_transition(self, error, sample):
        obs, nextobs, nextactset, rew, act, done = sample
        self.insert_idx = self.insert_idx % self.max_size
        self.obs[self.insert_idx] = obs
        self.rew[self.insert_idx] = rew
        self.act[self.insert_idx] = act

        self.nextobs[self.insert_idx] = nextobs
        self.nextactset[self.insert_idx] = nextactset
        self.done[self.insert_idx] = done

        self.insert_idx += 1
        self.size = min(self.size + 1, self.max_size)

    def is_ready(self):
        return self.size >= self.start_size

    def sample(self):
        assert self.is_ready()

        ids = np.random.randint(0, self.size, self.batch_size)
        return {
            'obs': self.obs[ids], 
            'act': self.act[ids], 
            'rew': self.rew[ids],
            'next_obs': self.nextobs[ids], 
            'next_actset': self.nextactset[ids],
            'done': self.done[ids]
        }



class PrioritizedReplay:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, max_size=50000, start_size=10, batch_size=32):
        self.tree = SumTree(max_size)
        self.capacity = max_size
        self.batch_size = batch_size
        self.start_size = start_size

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add_transition(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def is_ready(self):
        return self.tree.n_entries >= self.start_size

    def sample(self):
        assert self.is_ready()

        batch = {
            'obs': [], 
            'act': [], 
            'rew': [],
            'next_obs': [], 
            'next_actset': [],
            'done': []
        }
        idxs = []
        segment = self.tree.total() / self.batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = np.random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            idxs.append(idx)
            for key in batch.keys():
                batch[key].append(data[key])

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        
        batch["obs"] = np.asarray(batch["obs"], dtype=ecole.core.observation.NodeBipartiteObs)

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
