import numpy as np
import ecole


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

    def add_transition(self, obs, rew, act, nextobs, nextactset, done):
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
