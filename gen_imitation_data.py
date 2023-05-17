from tasks import make_instances
from utils import seed_stochastic_modules_globally
import ecole
import gzip
import pickle
from pathlib import Path
import os
import random
import threading
from threading import Thread
import queue
from queue import Queue
from tqdm import trange
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np


hydra.HYDRA_FULL_ERROR = 1

n_parallel_process = 16
scip_params = {
    'separating/maxrounds': 0,  # separate (cut) only at root node
    'presolving/maxrestarts': 0,  # disable solver restarts
    'limits/time': 60 * 60,  # solver time limit
    'timing/clocktype': 1,  # 1: CPU user seconds, 2: wall clock time
    # 'limits/gap': 3e-4,  # 0.03% relative primal-dual gap (default: 0.0)
    # 'limits/nodes': -1,
}


class ExploreThenStrongBranch:
    """
    This custom observation function class will randomly return either strong branching scores (expensive expert)
    or pseudocost scores (weak expert for exploration) when called at every node.
    """
    def __init__(self, expert_probability):
        self.expert_probability = expert_probability
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.strong_branching_function = ecole.observation.StrongBranchingScores()

    def before_reset(self, model):
        """
        This function will be called at initialization of the environments (before dynamics are reset).
        """
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)

    def extract(self, model, done):
        """
        Should we return strong branching or pseudocost scores at time node?
        """
        probabilities = [1-self.expert_probability, self.expert_probability]
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            return (self.strong_branching_function.extract(model, done), True)
        else:
            return (self.pseudocosts_function.extract(model, done), False)


def run_sampler(cfg, path, sample_n_queue, seed=0):
    instance_gen = make_instances(cfg, seed=seed)
    max_steps = cfg.experiment.max_steps

    env = ecole.environment.Branching(
        observation_function=(ExploreThenStrongBranch(expert_probability=0.3), ecole.observation.NodeBipartite()),
        scip_params=scip_params
    )

    n = sample_n_queue.get(timeout=5)

    while True:
        instance = next(instance_gen)
        observation, action_set, _, done, _ = env.reset(instance)
        t = 0
        while not done:
            (scores, save_samples), node_observation = observation
            action = action_set[scores[action_set].argmax()]

            if save_samples:
                data = [node_observation, action, action_set, scores]
                filename = f'{path}sample_{n}.pkl'
                with gzip.open(filename, 'wb') as f:
                    pickle.dump(data, f)
                sample_n_queue.task_done()

                try:
                    n = sample_n_queue.get(timeout=5)
                except queue.Empty:
                    curr_thread = threading.current_thread()
                    print(f'thread {curr_thread} finished')
                    return

            observation, action_set, _, done, _ = env.step(action)
            t += 1
            if max_steps is not None:
                if t >= max_steps:
                    # stop episode
                    break

def init_save_dir(path):
    _path = '../../../' + path
    Path(_path).mkdir(parents=True, exist_ok=True)
    return _path


@hydra.main(config_path='configs', config_name='config.yaml')
def run(cfg: DictConfig):
    # seeding
    if 'seed' not in cfg.experiment:
        cfg.experiment['seed'] = random.randint(0, 10000)
        seed_stochastic_modules_globally(cfg.experiment.seed)

    # print info
    print('\n\n\n')
    print(f'~'*80)
    print(f'Config:\n{OmegaConf.to_yaml(cfg)}')
    print(f'~'*80)

################
    path = cfg.experiment.path_to_load_imitation_data
    path = init_save_dir(path)
################

    print('Generating >={} samples in parallel on {} CPUs and saving to {}'.format(cfg.experiment.num_samples, n_parallel_process, os.path.abspath(path)))

    ecole.seed(cfg.experiment.seed)
    sample_n_queue = Queue(maxsize=64)

    threads = list()
    for i in range(n_parallel_process):
        process = Thread(target=run_sampler, args=(cfg, path, sample_n_queue, i))
        process.start()
        threads.append(process)

    for i in trange(cfg.experiment.num_samples):
        sample_n_queue.put(i)

    for process in threads:
        process.join()
    sample_n_queue.join()

if __name__ == '__main__':
    run()
