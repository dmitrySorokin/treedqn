import numpy as np
import ecole
from ecole.scip import Model
from omegaconf import DictConfig


def make_instances(cfg: DictConfig, seed=0):
    rng = ecole.core.RandomGenerator(seed)

    if cfg.instances.co_class == 'set_covering':
        instances = ecole.instance.SetCoverGenerator(rng=rng, **cfg.instances.co_class_kwargs)
    elif cfg.instances.co_class == 'combinatorial_auction':
        instances = ecole.instance.CombinatorialAuctionGenerator(rng=rng, **cfg.instances.co_class_kwargs)
    elif cfg.instances.co_class == 'maximum_independent_set':
        instances = ecole.instance.IndependentSetGenerator(rng=rng, **cfg.instances.co_class_kwargs)
    elif cfg.instances.co_class == 'ucfl':
        instances = generate_unsplittable_capacited_facility_location(seed=seed, **cfg.instances.co_class_kwargs)
    elif cfg.instances.co_class == 'knapsack':
        instances = generate_knapsack(seed=seed, **cfg.instances.co_class_kwargs)
    else:
        raise Exception(f'Unrecognised co_class {cfg.instances.co_class}')

    return instances


def generate_unsplittable_capacited_facility_location(seed, n_customers, n_facilities, ratio):
    """
    Generate a Capacited Facility Location problem with unsplittable supply.

    Saves it as a CPLEX LP file.

    Parameters
    ----------
    random : numpy.random.RandomState
        A random number generator.
    filename : str
        Path to the file to save.
    n_customers: int
        The desired number of customers.
    n_facilities: int
        The desired number of facilities.
    ratio: float
        The desired capacity / demand ratio.
    """
    rng = np.random.RandomState(seed)

    while True:
        c_x = rng.rand(n_customers)
        c_y = rng.rand(n_customers)

        f_x = rng.rand(n_facilities)
        f_y = rng.rand(n_facilities)

        demands = rng.randint(5, 35+1, size=n_customers)
        capacities = rng.randint(10, 160+1, size=n_facilities)
        fixed_costs = rng.randint(100, 110+1, size=n_facilities) * np.sqrt(capacities) \
                + rng.randint(90+1, size=n_facilities)
        fixed_costs = fixed_costs.astype(int)

        total_demand = demands.sum()
        total_capacity = capacities.sum()

        # adjust capacities according to ratio
        capacities = capacities * ratio * total_demand / total_capacity
        capacities = capacities.astype(int)
        total_capacity = capacities.sum()

        # transportation costs
        trans_costs = np.sqrt(
                (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 \
                + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2) * 10 * demands.reshape((-1, 1))

        # write problem
        with open(f'ucfl_task_{seed}.lp', 'w') as file:
            file.write("minimize\nobj:")
            file.write("".join([f" +{trans_costs[i, j]} x_{i+1}_{j+1}" for i in range(n_customers) for j in range(n_facilities)]))
            file.write("".join([f" +{fixed_costs[j]} y_{j+1}" for j in range(n_facilities)]))

            file.write("\n\nsubject to\n")
            for i in range(n_customers):
                file.write(f"demand_{i+1}:" + "".join([f" -1 x_{i+1}_{j+1}" for j in range(n_facilities)]) + f" <= -1\n")
            for j in range(n_facilities):
                file.write(f"capacity_{j+1}:" + "".join([f" +{demands[i]} x_{i+1}_{j+1}" for i in range(n_customers)]) + f" -{capacities[j]} y_{j+1} <= 0\n")

            # optional constraints for LP relaxation tightening
            file.write("total_capacity:" + "".join([f" -{capacities[j]} y_{j+1}" for j in range(n_facilities)]) + f" <= -{total_demand}\n")
            for i in range(n_customers):
                for j in range(n_facilities):
                    file.write(f"affectation_{i+1}_{j+1}: +1 x_{i+1}_{j+1} -1 y_{j+1} <= 0")

            file.write("\nbinary\n")
            file.write("".join([f" y_{j+1}" for j in range(n_facilities)]))
            file.write("".join([f" x_{i+1}_{j+1}" for i in range(n_customers) for j in range(n_facilities)]))

        task = Model.from_file(f'ucfl_task_{seed}.lp')
        yield task



def generate_knapsack(
        seed,
        number_of_items, number_of_knapsacks,
        min_range=10, max_range=20, scheme='subset-sum'):
    """
    Generate a Multiple Knapsack problem following a scheme among those found in section 2.1. of
        Fukunaga, Alex S. (2011). A branch-and-bound algorithm for hard multiple knapsack problems.
        Annals of Operations Research (184) 97-119.
    Saves it as a CPLEX LP file.
    Parameters
    ----------
    number_of_items : int
        The number of items.
    number_of_knapsacks : int
        The number of knapsacks.
    filename : str
        Path to the file to save.
    random : numpy.random.RandomState
        A random number generator.
    min_range : int, optional
        The lower range from which to sample the item weights. Default 10.
    max_range : int, optional
        The upper range from which to sample the item weights. Default 20.
    scheme : str, optional
        One of 'uncorrelated', 'weakly correlated', 'strongly corelated', 'subset-sum'. Default 'weakly correlated'.
    """
    rng = np.random.RandomState(seed)

    while True:
        weights = rng.randint(min_range, max_range, number_of_items)

        if scheme == 'uncorrelated':
            profits = rng.randint(min_range, max_range, number_of_items)

        elif scheme == 'weakly correlated':
            profits = np.apply_along_axis(
                lambda x: rng.randint(x[0], x[1]),
                axis=0,
                arr=np.vstack([
                    np.maximum(weights - (max_range-min_range), 1),
                    weights + (max_range-min_range)]))

        elif scheme == 'strongly correlated':
            profits = weights + (max_range - min_range) / 10

        elif scheme == 'subset-sum':
            profits = weights

        else:
            raise NotImplementedError

        capacities = np.zeros(number_of_knapsacks, dtype=int)
        capacities[:-1] = rng.randint(0.4 * weights.sum() // number_of_knapsacks,
                                            0.6 * weights.sum() // number_of_knapsacks,
                                            number_of_knapsacks - 1)
        capacities[-1] = 0.5 * weights.sum() - capacities[:-1].sum()

        with open(f'knapsack_task_{seed}.lp', 'w') as file:
            file.write("maximize\nOBJ:")
            for knapsack in range(number_of_knapsacks):
                for item in range(number_of_items):
                    file.write(f" +{profits[item]} x{item+number_of_items*knapsack+1}")

            file.write("\n\nsubject to\n")
            for knapsack in range(number_of_knapsacks):
                variables = "".join([f" +{weights[item]} x{item+number_of_items*knapsack+1}"
                                    for item in range(number_of_items)])
                file.write(f"C{knapsack+1}:" + variables + f" <= {capacities[knapsack]}\n")

            for item in range(number_of_items):
                variables = "".join([f" +1 x{item+number_of_items*knapsack+1}"
                                    for knapsack in range(number_of_knapsacks)])
                file.write(f"C{number_of_knapsacks+item+1}:" + variables + " <= 1\n")

            file.write("\nbinary\n")
            for knapsack in range(number_of_knapsacks):
                for item in range(number_of_items):
                    file.write(f" x{item+number_of_items*knapsack+1}")
        
        yield Model.from_file(f'knapsack_task_{seed}.lp')


def gen_co_name(co_class, co_class_kwargs):
    _str = f'{co_class}'
    for key, val in co_class_kwargs.items():
        _str += f'_{key}_{val}'
    return _str
