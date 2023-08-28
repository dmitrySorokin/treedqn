from agent import DQNAgent, ImitationAgent, ReinforceAgent, FMCTSAgent
from env import EcoleBranching
from tasks import gen_co_name

import ecole
import glob
import numpy as np
import hydra
from omegaconf import DictConfig
import os
from multiprocessing.pool import ThreadPool
from typing import Iterable, Callable, NamedTuple
from functools import partial
from numpy.random import default_rng
from csv import DictWriter
from time import perf_counter


def get_randombrancher(seed: int = None) -> Callable:
    """Return the policy of random branching"""
    rng = default_rng(seed)

    def _randombrancher(env: ecole.environment.Branching) -> Callable:
        """Get the random branching rule for env"""

        def _do_branch(obs: ..., act_set: ..., deterministic: bool = True) -> int:
            """Randomly pick which variable to branch on"""
            return int(rng.choice(act_set))

        return _do_branch

    return _randombrancher


def get_strongbrancher() -> Callable:
    """Return the strong branching policy"""

    def _strongbrancher(env: ecole.environment.Branching) -> Callable:
        """Get the strong branching policy for env"""
        sbf = ecole.observation.StrongBranchingScores()

        def _do_branch(obs: ..., act_set: ..., deterministic: bool = True) -> int:
            """Decide which variable to branch on using the SB heuristic"""
            scores = sbf.extract(env.model, False)[act_set]
            return act_set[np.argmax(scores)]

        return _do_branch

    return _strongbrancher


def get_dqnbrancher(ckpt: str, device: str = "cpu") -> Callable:
    """Load a DQN agent from a checkpoint and return a brancher"""
    assert os.path.isfile(ckpt)

    agent = DQNAgent(device=device, epsilon=0)
    agent.load(ckpt)

    def _dqnbrancher(env: ecole.environment.Branching) -> Callable:
        """Get a dqn-based branching policy for env"""

        def _do_branch(obs: ..., act_set: ..., deterministic: bool = True) -> int:
            """Decide on the branching variable with a graph DQN"""
            agent.eval()
            act = agent.act(obs, act_set, deterministic=deterministic)
            return act

        return _do_branch

    return _dqnbrancher


def get_fmctsbrancher(ckpt: str, device: str = "cpu") -> Callable:
    """Load a FMCTS agent from a checkpoint and return a brancher"""
    assert os.path.isfile(ckpt)

    agent = FMCTSAgent(device=device, epsilon=0)
    agent.load(ckpt)

    def _brancher(env: ecole.environment.Branching) -> Callable:
        """Get a FMCTS-based branching policy for env"""

        def _do_branch(obs: ..., act_set: ..., deterministic: bool = True) -> int:
            """Decide on the branching variable with a graph DQN"""
            agent.eval()
            act = agent.act(obs, act_set, deterministic=deterministic)
            return act

        return _do_branch

    return _brancher


def get_ilbrancher(cfg, ckpt: str, device: str = "cpu") -> Callable:
    """Load a IL agent from a checkpoint and return a brancher"""
    assert os.path.isfile(ckpt)

    agent = ImitationAgent(device=device)
    agent.load(ckpt)

    def _ilbrancher(env: ecole.environment.Branching) -> Callable:
        """Get a il-based branching policy for env"""

        def _do_branch(obs: ..., act_set: ..., deterministic: bool = True) -> int:
            """Decide on the branching variable with a graph DQN"""
            agent.eval()
            act = agent.act(obs, act_set, deterministic=deterministic)
            return act

        return _do_branch

    return _ilbrancher


def get_reinforcebrancher(cfg, ckpt: str, device: str = "cpu") -> Callable:
    """Load a IL agent from a checkpoint and return a brancher"""
    assert os.path.isfile(ckpt)

    agent = ReinforceAgent(device=device)
    agent.load(ckpt)
    agent.eval()

    def _brancher(env: ecole.environment.Branching) -> Callable:
        """Get a il-based branching policy for env"""

        def _do_branch(obs: ..., act_set: ..., deterministic: bool = True) -> int:
            """Decide on the branching variable with a graph DQN"""
            act = agent.act(obs, act_set)
            return act

        return _do_branch

    return _brancher


class Job(NamedTuple):
    alias: str
    instance: str
    seed: int
    name: str


def evaluate_one(
    branchers: dict[str, Callable],
    j: Job,
    *,
    stop: Callable[[], bool] = (lambda: False),
) -> dict:

    env = EcoleBranching(None, params={"limits/nodes": 200000, "limits/time": 60 * 60 * 60})
    env.seed(j.seed)

    pick = branchers[j.name](env)

    n_steps, f_wtime = 0, perf_counter()
    obs, act_set, _, done, _ = env.reset_basic(j.instance)  # XXX `.reset` is too smart
    while not done and not stop():
        act = pick(obs, act_set, deterministic=True)
        obs, act_set, _, done, _ = env.step(act)
        n_steps += 1

    f_wtime = perf_counter() - f_wtime

    # Manually check if SCIP encountered a sigint
    m = env.model.as_pyscipopt()
    if m.getStatus() == "userinterrupt":
        raise KeyboardInterrupt from None

    # collect the stats manually (extended fields)
    return dict(
        name=j.name,
        seed=j.seed,
        instance=j.alias,
        n_interactions=n_steps,
        n_lps=m.getNLPs(),
        num_nodes=m.getNNodes(),
        n_lpiter=m.getNLPIterations(),
        f_gap=m.getGap(),
        f_soltime=m.getSolvingTime(),  # XXX meaningless with n_jobs > 1
        s_status=m.getStatus(),
        f_walltime=f_wtime,
    )


def main(
    branchers: dict[str, Callable],
    folder: str,
    seeds: tuple[int] = (123,),
    n_workers: int = 1,
    *,
    filter: str = " *.lp",
) -> Iterable:
    # enumerate "*.lp" files in a given folder
    folder = glob.os.path.abspath(folder)
    files = glob.glob(glob.os.path.join(folder, filter), recursive=False)

    prefix = os.path.commonpath(files)

    # spawn the jobs and run evaluation
    jobs = []
    for seed in seeds:
        for name in branchers:
            jobs.extend(Job(f.removeprefix(prefix), f, seed, name) for f in files)

    with ThreadPool(n_workers) as p:
        fn = partial(evaluate_one, branchers)
        yield from p.imap_unordered(fn, jobs, chunksize=1)


@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def evaluate(cfg: DictConfig):
    # the seeds we use for the env during eval
    seeds = (0, 1, 2, 3, 4)

    n_workers = int(cfg.get("j", 16))
    assert n_workers > 0

    # prepare the co problem instance source
    co_name = gen_co_name(cfg.instances.co_class, cfg.instances.co_class_kwargs)
    folder = os.path.abspath(cfg.get("folder", f"test_instances/{co_name}"))

    # prepare the branchers
    if cfg.agent.name == "strong":
        brancher = get_strongbrancher()

    elif cfg.agent.name == "dqn":
        device = cfg.get("device", "cpu")
        ckpt = os.path.abspath(cfg.agent.checkpoint)

        print(f"Loading `{ckpt}` to {device}")
        brancher = get_dqnbrancher(ckpt, device)

    elif cfg.agent.name == "fmcts":
        device = cfg.get("device", "cpu")
        ckpt = os.path.abspath(cfg.agent.checkpoint)

        print(f"Loading `{ckpt}` to {device}")
        brancher = get_fmctsbrancher(ckpt, device)


    elif cfg.agent.name == "random":
        brancher = get_randombrancher(seed=None)  # XXX seed is deliberately not fixed

    elif cfg.agent.name == "il":
        device = cfg.get("device", "cpu")
        ckpt = os.path.abspath(cfg.agent.checkpoint)

        print(f"Loading `{ckpt}` to {device}")
        brancher = get_ilbrancher(cfg, ckpt, device)

    elif cfg.agent.name == "reinforce":
        device = cfg.get("device", "cpu")
        ckpt = os.path.abspath(cfg.agent.checkpoint)

        print(f"Loading `{ckpt}` to {device}")
        brancher = get_reinforcebrancher(cfg, ckpt, device)

    else:
        raise NotImplementedError(f"Unknown agent name {cfg.agent.name}")

    # yep, only one brancher
    branchers = {cfg.agent.name: brancher}

    # prepare the output folder and fill the csv file
    out_dir = os.path.abspath(cfg.get("out_dir", f"results/{co_name}"))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Saving to `{out_dir}`")

    csv = os.path.join(out_dir, f"{cfg.agent.name}{cfg.agent.suffix}.csv")
    with open(csv, "wt", newline="") as f:
        writer = DictWriter(
            f,
            fieldnames=[
                "num",
                "name",
                "seed",
                "instance",
                "n_interactions",
                "n_lps",
                "num_nodes",
                "n_lpiter",
                "f_gap",
                "f_soltime",
                "s_status",
                "f_walltime",
            ],
        )

        it = main(branchers, folder, seeds=seeds, n_workers=n_workers, filter="*.lp")

        writer.writeheader()
        for j, result in enumerate(it):
            writer.writerow({"num": j, **result})
            f.flush()

            if (j % 25) == 0:
                print(
                    "\n      name         seed     status   "
                    "   steps    nodes      lps   walltime  instance"
                )
            print(
                "{j:>5d} {name:<12} {seed:<8} {s_status:<8} {n_interactions:>8}"
                " {num_nodes:>8} {n_lps:>8} {f_walltime:>8.2f}s.  {instance}"
                "".format(j=j, **result)
            )


if __name__ == "__main__":
    # python eval.py --config-name "combinatorial_auction" ++j=8 ++out_dir="./test"
    #  ++agent.name="strong"
    #  ++agent.name="dqn" "~agent.checkpoint"
    evaluate()
