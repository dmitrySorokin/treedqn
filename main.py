import hydra
import numpy as np
from omegaconf import DictConfig
from env import EcoleBranching
from tasks import make_instances
from agent import FMCTSAgent, ReplayBuffer
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
from torch import multiprocessing as mp
from tasks import gen_co_name
import time


class EvalProcess(mp.Process):
    def __init__(self, config, in_queue, out_queue):
        super().__init__()
        self.cfg = config
        self.in_queue = in_queue
        self.out_queue = out_queue

    def run(self):
        env = EcoleBranching(None)
        agent = FMCTSAgent(device=self.cfg.experiment.device, epsilon=0)
        co_name = gen_co_name(self.cfg.instances.co_class, self.cfg.instances.co_class_kwargs)
        folder = f'../../../validate_instances/{co_name}'
        tasks = [(os.path.join(folder, f'instance_{j + 1}.lp'), seed) for j in range(20) for seed in range(5)]
        stop = False
        while not stop:
            if self.in_queue.empty():
                time.sleep(60)
                continue
            (checkpoint, episode, stop) = self.in_queue.get()

            agent.load(checkpoint)
            n_nodes = []
            for file, seed in tasks:
                env.seed(seed)
                obs, act_set, _, done, _ = env.reset_basic(file)
                while not done:
                    act = agent.act(obs, act_set, deterministic=True)
                    obs, act_set, _, done, _ = env.step(act)
                n_nodes.append(env.model.as_pyscipopt().getNNodes())
            geomean = np.exp(np.mean(np.log(n_nodes)))
            self.out_queue.put((geomean, episode))


def rollout(env, agent, replay_buffer, max_tree_size=1000):
    obs, act_set, (returns, children_ids), done, info = env.reset()
    
    traj_obs, traj_ret, traj_act, traj_actset, traj_done = [], [], [], [], []
    while not done:
        action = agent.act(obs, act_set, deterministic=False)
        traj_obs.append(obs)
        traj_act.append(action)
        traj_actset.append(act_set)
        obs, act_set, (returns, children_ids), done, info = env.step(action)
        traj_done.append(done)

    traj_nextobs, traj_nextactset = [], []
    for ret, children in zip(returns, children_ids):
        traj_nextobs.append([traj_obs[c] for c in children])
        traj_nextactset.append([traj_actset[c] for c in children])
        traj_ret.append(ret)

    assert len(traj_obs) == len(traj_nextobs)
    tree_size = len(traj_obs)
    # ids = np.random.choice(range(tree_size), min(tree_size, max_tree_size), replace=False)
    ids = list(range(min(tree_size, max_tree_size)))
    traj_obs = np.asarray(traj_obs)[ids]
    traj_ret = np.asarray(traj_ret)[ids]
    traj_act = np.asarray(traj_act)[ids]
    
    traj_nextobs = np.array(traj_nextobs, dtype=list)[ids]
    traj_nextactset = np.array(traj_nextactset, dtype=list)[ids]
    traj_done = np.asarray(traj_done)[ids]


    for transition in zip(traj_obs, traj_ret, traj_act, traj_nextobs, traj_nextactset, traj_done):
        replay_buffer.add_transition(*transition, tree_size)

    return len(ids), info


@hydra.main(config_path='configs', config_name='config.yaml')
def main(cfg: DictConfig):
    mp.set_start_method('spawn')
    writer = SummaryWriter(os.getcwd())

    env = EcoleBranching(make_instances(cfg, cfg.experiment.seed), dfs=True)
    env.seed(cfg.experiment.seed)

    agent = FMCTSAgent(device=cfg.experiment.device, epsilon=1)
    agent.train()

    replay_buffer = ReplayBuffer(
        max_size=cfg.experiment.buffer_max_size,
        start_size=cfg.experiment.buffer_start_size
    )

    pbar = tqdm(total=replay_buffer.start_size, desc='init')
    while not replay_buffer.is_ready():
        num_obs, _ = rollout(env, agent, replay_buffer)
        pbar.update(num_obs)
    pbar.close()

    pbar = tqdm(total=cfg.experiment.num_episodes, desc='train')
    update_id = 0
    episode_id = 0
    epsilon_min = 0.01
    decay_steps = 100000

    in_queue, out_queue = mp.Queue(), mp.Queue()
    evaler = EvalProcess(cfg, in_queue, out_queue)
    evaler.start()

    while episode_id <= pbar.total:
        num_obs, info = rollout(env, agent, replay_buffer)
        writer.add_scalar('num_nodes', info['num_nodes'], episode_id)
        writer.add_scalar('lp_iterations', info['lp_iterations'], episode_id)
        writer.add_scalar('solving_time', info['solving_time'], episode_id)
        writer.add_scalar('epsilon', agent.epsilon, episode_id)

        print(episode_id, info['num_nodes'])
        
        episode_loss = []
        for i in range(num_obs):
            episode_loss.append(agent.update(update_id, replay_buffer.sample()))
            update_id += 1
        
        writer.add_scalar('loss', np.mean(episode_loss), episode_id)

        chkpt = os.getcwd() + f'/checkpoint_{episode_id}.pkl'
        agent.save(chkpt)

        if episode_id % cfg.experiment.eval_freq == 0 or episode_id == cfg.experiment.num_episodes:
            chkpt = os.getcwd() + f'/checkpoint_{episode_id}.pkl'
            agent.save(chkpt)
            in_queue.put((chkpt, episode_id, episode_id == cfg.experiment.num_episodes))

        episode_id += 1
        epsilon = 1. - (1. - epsilon_min) / decay_steps * update_id
        agent.epsilon = max(epsilon_min, epsilon)

        while not out_queue.empty():
            writer.add_scalar('eval', *out_queue.get_nowait())


        pbar.update(1)
    evaler.join()
    pbar.close()

    while not out_queue.empty():
        writer.add_scalar('eval', *out_queue.get_nowait())


if __name__ == '__main__':
    main()
