from .reward import Reward
import ecole


class DFSBranchingDynamics(ecole.dynamics.BranchingDynamics):
    """
    Custom branching environment that changes the node strategy to DFS when training.
    """
    def reset_dynamics(self, model, training, *args, **kwargs):
        pyscipopt_model = model.as_pyscipopt()
        if training:
            # Set the dfs node selector as the least important
            pyscipopt_model.setParam(f"nodeselection/dfs/stdpriority", 666666)
            pyscipopt_model.setParam(f"nodeselection/dfs/memsavepriority", 666666)
        else:
            # Set the dfs node selector as the most important
            pyscipopt_model.setParam(f"nodeselection/dfs/stdpriority", 0)
            pyscipopt_model.setParam(f"nodeselection/dfs/memsavepriority", 0)

        return super().reset_dynamics(model, *args, **kwargs)


class EcoleBranching(ecole.environment.Branching):
    __Dynamics__ = DFSBranchingDynamics

    def __init__(self, instance_gen, params=None, dfs=False):
        # init default rewards
        reward_function = Reward()

        information_function = { 
            "num_nodes": ecole.reward.NNodes().cumsum(),
            "lp_iterations": ecole.reward.LpIterations().cumsum(),
            "solving_time": ecole.reward.SolvingTime().cumsum(),
        }

        scip_params = {
            "separating/maxrounds": 0,  # separate (cut) only at root node
            "presolving/maxrestarts": 0,  # disable solver restarts
            "limits/time": 60 * 10,  # solver time limit
            "timing/clocktype": 2,  # 1: CPU user seconds, 2: wall clock time
        }

        if params:
            scip_params.update(params)

        super(EcoleBranching, self).__init__(
            observation_function=ecole.observation.NodeBipartite(),
            information_function=information_function,
            reward_function=reward_function,
            scip_params=scip_params,
            pseudo_candidates=False,
        )

        self.instance_gen = instance_gen
        self.dfs = dfs

    def reset_basic(self, instance):
        return super().reset(instance, training=self.dfs)

    def reset(self):
        for instance in self.instance_gen:
            obs, act_set, reward, done, info = super().reset(instance.copy_orig(), training=self.dfs)
            if not done:
                info["instance"] = instance
                return obs, act_set, reward, done, info
        raise StopIteration
    
