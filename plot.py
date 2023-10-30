#!/bin/python3

import numpy as np
from numpy import ndarray
import pandas as pd
from matplotlib import pyplot as plt
from argparse import ArgumentParser
import os


def geomean(data):
    return np.exp(np.mean(np.log(data)))


def pp_curve(*, x: ndarray, y: ndarray, ystd: ndarray, num: int = None):
    """Build threshold-parameterized pipi curve."""
    # sort each sample for fast O(\log n) eCDF queries by `searchsorted`
    x, y = np.sort(x), np.sort(y)

    # pool sorted samples to get thresholds
    xy = np.concatenate((x, y))
    if num is None:
        # finest detail thresholds: sort the pooled samples (sorted
        #  arrays can be merged in O(n), but it turns out numpy does
        #  not have the procedure)
        xy.sort()

    else:
        # coarsen by finding threshold grid in the pooled sample, that
        #  is equispaced after being transformed by the empirical cdf.
        xy = np.quantile(xy, np.linspace(0, 1, num=num), interpolation='linear')

    # add +ve/-ve inf end points to the parameter value sequence
    xy = np.r_[-np.inf, xy, +np.inf]

    # we build the pp-curve the same way as we build the ROC curve:
    #  by parameterizing with the a monotonic threshold sequence
    #    pp: v \mapsto (\hat{F}_x(v), \hat{F}_y(v))
    #  where \hat{F}_S(v) = \frac1{n_S} \sum_j 1_{S_j \leq v}
    p = np.searchsorted(x, xy) / len(x)
    q = np.searchsorted(y, xy) / len(y)

    return p, q


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--key', default='num_nodes', choices=['num_nodes', 'lp_iterations', 'solving_time'])
    parser.add_argument('--nodelimit', default=200000, type=int)
    args = parser.parse_args()


    params = {
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
    }
    plt.rcParams.update(params)

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    key, path = args.key, args.path
    strong = pd.read_csv(f'{path}/strong.csv')[key].to_numpy()

    results = []

    colors = {
        'strong': 'tab:blue',
        'dqn': 'tab:orange',
        'reinforce': 'tab:green',
        'fmcts': 'black',
        'il': 'tab:red'
    }

    ignore = {
        'dqn_dfs.csv',
        'dqn_mse.csv',
        'dqn_dfslong.csv',
        'scavuzzo_objlim.csv',
        'scavuzzo.csv',
        'dqn_net.csv'
    }

    for fname in os.listdir(path):
        if not fname.endswith('.csv'):
            continue

        data = pd.read_csv(f'{path}/' + fname)
        names = list(set(data['name']))
        assert len(names) == 1, f'{fname}'

        stds, values = [], []
        for _, taskgroup in data.groupby('instance'):
            group = np.minimum(taskgroup[key].to_numpy(), args.nodelimit)
            stds.append(np.std(group) / np.mean(group))
            values.extend(group)

        n_timeout = len(data[data['s_status'] == 'timelimit'])
        n_nodelimit = len(data[data['num_nodes'] >= args.nodelimit])
        results.append((*names, fname, values, stds, n_timeout, n_nodelimit))

        # plt.hist(data, bins=100, log=True)
        # plt.title(fname)
        # plt.savefig(f'hist_{fname[:-4]}.pdf')
        # plt.close()

    print(f"{'name':<40} {'tot':<10} {'geomean':<10} {'mean':<10} {'std':<10} {'timeout':<10} {'nodelimit':<10}")
    print('-' * 80)
    for (name, fname, data, stds, n_timeout, n_nodelimit) in sorted(results, key=lambda item: geomean(item[2])):
        full_name = f'{name}({fname})'
        print(f'{full_name:<40} {len(data):<10.2f} {geomean(data):<10.2f} {np.mean(data):<10.2f} {np.mean(stds) * 100:<10.2f} {n_timeout:<10.2f} {n_nodelimit:<10.2f}')

        if fname in ignore:
            continue

        u, p = pp_curve(x=strong, y=data, ystd=stds)
        ax.plot(u, p, label=name, color=colors[name], linewidth=2.0)

        # ax.set_yticklabels([])
        # ax.set_xticklabels([])

    ax.plot((0, 1), (0, 1), c="k", zorder=10, alpha=0.25)
    ax.set_xlim(-0.025, 1.025)
    ax.set_ylim(-0.025, 1.025)
    ax.set_aspect(1.)
    # ax.set_title('Comb.Auct.')

    # plt.legend()
    # plt.tight_layout(rect=[0.3, 0.03, 1.1, 1.3])
    # plt.tight_layout()
    plt.margins(x=0, y=0)
    plt.savefig(f'{path}/pp_{key}.pdf', bbox_inches='tight')
    plt.show()
