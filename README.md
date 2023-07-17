# TreeDQN: Learning to minimize Branch-and-Bound tree

This repository is the official implementation of [TreeDQN: Learning to minimize Branch-and-Bound tree](https://arxiv.org/abs/2306.05905). 


<p float="center">
  <img src="gifs/cauct_1.gif" width="400"/>
</p>

## Setup environment


```bash
# pull docker image
docker pull idono/rlbnb:release

# run docker container
docker run -dit --gpus all --shm-size=10g --name rlbnb rlbnb:release /bin/bash

# enter docker container
docker exec -it rlbnb /bin/bash

# work with TreeDQN
git clone https://github.com/dmitrySorokin/treedqn.git
conda activate bb

# work with baseline rl2branch
git clone https://github.com/lascavana/rl2branch.git
conda activate rl2branch
```

## Training

To train the RL agent, run this commands:

```bash
# generate validation data
python gen_instances.py --config-name <cfg from configs>

# run training
python main.py --config-name <cfg from configs>
```

To train the IL agent, run this commands:

```bash
# generate training data
python gen_imitation_data.py --config-name <cfg from configs>

# run training
python il_train.py --config-name <cfg from configs>
```

## Evaluation

To evaluate the agent, run:

```bash

python eval.py --config-name <cfg from configs> agent.name={agent_name}
```
* agent_name: strong, dqn, il, random
* results will be saved in results/{task_name}/{agent_name}.csv


## Pre-trained Models

Pretrained weights for IL, TreeDQN and REINFORCE agents are in models/ dir.

## P-P plots
To plot results, run:
```bash
python plot.py results/<task name>
```

## Results

Geometric mean of tree sizes (lower is better):

|Model | Comb.Auct | Set Cover | Max.Ind.Set. | Facility Loc. | Mult.Knap |
|------|-----------|-----------|--------------|---------------|----------------|
|Strong Branching | 48 $\pm$ 14\% | 43 $\pm$ 8\% | 40 $\pm$ 36\% | 294 $\pm$ 53\% | 700 $\pm$ 116\% |
|IL | 56 $\pm$ 12\% | 53 $\pm$ 9\% | 42 $\pm$ 32\% | 323 $\pm$ 46\% | 670 $\pm$ 120\% |
|TreeDQN | **62 $\pm$ 15\%** | **57 $\pm$ 11\%** | **47 $\pm$ 41\%** | **392 $\pm$ 49\%** | **303 $\pm$ 88\%**|
|REINFORCE | 93 $\pm$ 18\% | 249 $\pm$ 23\% | 75 $\pm$ 39\% | 521 $\pm$ 50\% | 308 $\pm$ 103\%|



## Contributing

Submit Github issue if you have any questions or want to contribute. 
