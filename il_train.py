import torch_geometric
from agent import ImitationAgent
from utils import GraphDataset, seed_stochastic_modules_globally
import glob
import numpy as np
import os
import random
import pathlib

from tensorboardX import SummaryWriter

import hydra
from omegaconf import DictConfig
hydra.HYDRA_FULL_ERROR = 1


def init_save_dir(self, path='.', agent_name=None):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


@hydra.main(config_path='configs', config_name='config.yaml')
def run(cfg: DictConfig):
    # seeding
    if 'seed' not in cfg.experiment:
        cfg.experiment['seed'] = random.randint(0, 10000)
    seed_stochastic_modules_globally(cfg.experiment.seed)

    writer = SummaryWriter(os.getcwd())

    # initialise imitation agent
    agent = ImitationAgent(device=cfg.experiment.device)
    agent.train()
    print('Initialised imitation agent.')

    # get paths to labelled training and validation data
    path = '../../../' + cfg.experiment.path_to_load_imitation_data

    print(f'Loading imitation data from {path}...')
    if not os.path.isdir(path):
        raise Exception(f'Path {path} does not exist')
    sample_files = np.array(glob.glob(path+'*.pkl'))[:cfg.experiment.num_samples]
    train_files = sample_files[:int(0.83*len(sample_files))]
    valid_files = sample_files[int(0.83*len(sample_files)):]

    # init training and validaton data loaders
    train_data = GraphDataset(train_files)
    train_loader = torch_geometric.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    valid_data = GraphDataset(valid_files)
    valid_loader = torch_geometric.data.DataLoader(valid_data, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    print('Initialised training and validation data loaders.')

    for epoch in range(cfg.experiment.num_epochs):
        train_loss, val_loss = 0, 0
        train_iters, val_iters = 0, 0

        agent.train()
        for train_batch in train_loader:
            train_batch = train_batch.to(agent.device)
            train_loss += agent.update(train_batch)
            train_iters += 1
        train_loss /= train_iters

        agent.eval()
        for val_batch in valid_loader:
            val_batch = val_batch.to(agent.device)
            val_loss += agent.validate(val_batch)
            val_iters += 1
        val_loss /= val_iters

        if epoch % cfg.experiment.epoch_log_frequency == 0:
            writer.add_scalar('epoch/train_loss', train_loss, epoch)
            writer.add_scalar('epoch/val_loss', val_loss, epoch)

        if epoch % cfg.experiment.checkpoint_log_frequency == 0:
            agent.save(os.getcwd(), epoch)

        print("Epoch: %d\tTrain loss: %.12f\tVal loss: %.12f" % (epoch, train_loss, val_loss))


if __name__ == '__main__':
    run()
