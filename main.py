import hydra
import numpy as np
from omegaconf import DictConfig
from env import EcoleBranching
from tasks import make_instances
from agent import DQNAgent, ReplayBuffer
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
        agent = DQNAgent(device=self.cfg.experiment.device, epsilon=0)
        co_name = gen_co_name(self.cfg.instances.co_class, self.cfg.instances.co_class_kwargs)
        update_id, max_updates = -1, self.cfg.experiment.num_updates
        folder = f'../../../validate_instances/{co_name}'
        while True:
            if self.in_queue.empty():
                time.sleep(60)
                continue
            (checkpoint, update_id) = self.in_queue.get()
            agent.load(checkpoint)
            n_nodes = []
            for task in os.listdir(folder):
                if not task.endswith('.lp'):
                    continue
                env.seed(0)
                obs, act_set, _, done, _ = env.reset_basic(os.path.join(folder, task))
                while not done:
                    act = agent.act(obs, act_set, deterministic=True)
                    obs, act_set, _, done, _ = env.step(act)
                n_nodes.append(env.model.as_pyscipopt().getNNodes())
            geomean = np.exp(np.mean(np.log(n_nodes)))
            self.out_queue.put((geomean, update_id))

            if update_id == max_updates:
                break


def rollout(env, agent, replay_buffer, max_tree_size=1000):
    obs, act_set, children_ids, done, info = env.reset()
    
    traj_obs, traj_rew, traj_act, traj_actset, traj_done = [], [], [], [], []
    while not done:
        action = agent.act(obs, act_set, deterministic=False)
        traj_obs.append(obs)
        traj_act.append(action)
        traj_actset.append(act_set)
        obs, act_set, children_ids, done, info = env.step(action)
        traj_done.append(done)

    traj_nextobs, traj_nextactset = [], []
    for children in children_ids:
        traj_nextobs.append([traj_obs[c] for c in children])
        traj_nextactset.append([traj_actset[c] for c in children])
        traj_rew.append(-1)

    assert len(traj_obs) == len(traj_nextobs)
    tree_size = len(traj_obs)
    # ids = np.random.choice(range(tree_size), min(tree_size, max_tree_size), replace=False)
    ids = list(range(min(tree_size, max_tree_size)))
    traj_obs = np.asarray(traj_obs)[ids]
    traj_rew = np.asarray(traj_rew)[ids]
    traj_act = np.asarray(traj_act)[ids]
    
    traj_nextobs = np.array(traj_nextobs, dtype=list)[ids]
    traj_nextactset = np.array(traj_nextactset, dtype=list)[ids]
    traj_done = np.asarray(traj_done)[ids]


    for transition in zip(traj_obs, traj_rew, traj_act, traj_nextobs, traj_nextactset, traj_done):
        replay_buffer.add_transition(*transition)

    return len(ids), info


@hydra.main(config_path='configs', config_name='config.yaml')
def main(cfg: DictConfig):
    mp.set_start_method('spawn')
    writer = SummaryWriter(os.getcwd())

    env = EcoleBranching(make_instances(cfg, cfg.experiment.seed), dfs=True)
    env.seed(cfg.experiment.seed)

    agent = DQNAgent(device=cfg.experiment.device, epsilon=1)
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

    pbar = tqdm(total=cfg.experiment.num_updates, desc='train')
    update_id = 0
    episode = 0
    epsilon_min = 0.01

    in_queue, out_queue = mp.Queue(), mp.Queue()
    evaler = EvalProcess(cfg, in_queue, out_queue)
    evaler.start()

    while update_id < pbar.total:
        num_obs, info = rollout(env, agent, replay_buffer)
        writer.add_scalar('episode/num_nodes', info['num_nodes'], episode)
        writer.add_scalar('episode/lp_iterations', info['lp_iterations'], episode)
        writer.add_scalar('episode/solving_time', info['solving_time'], episode)
        print(episode, info['num_nodes'])
        episode += 1
        for i in range(num_obs):
            loss = agent.update(update_id, replay_buffer.sample())
            writer.add_scalar('update/loss', loss, update_id)
            writer.add_scalar('update/epsilon', agent.epsilon, update_id)
            update_id += 1
            
            epsilon = 1. - (1. - epsilon_min) / pbar.total * update_id
            agent.epsilon = max(epsilon_min, epsilon)

            if update_id % cfg.experiment.eval_freq == 0:
                chkpt = os.getcwd() + f'/checkpoint_{update_id}.pkl'
                agent.save(chkpt)
                in_queue.put((chkpt, update_id))

        chkpt = os.getcwd() + f'/checkpoint_{update_id}.pkl'
        agent.save(chkpt)

        while not out_queue.empty():
            writer.add_scalar('update/eval', *out_queue.get_nowait())


        pbar.update(num_obs)
    evaler.join()
    pbar.close()


if __name__ == '__main__':
    main()
