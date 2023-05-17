import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np
import gzip
import pickle
import ecole
import random


def seed_stochastic_modules_globally(default_seed=0,
                                     numpy_seed=None,
                                     random_seed=None,
                                     torch_seed=None,
                                     ecole_seed=None):
    '''Seeds any stochastic modules so get reproducible results.'''
    if numpy_seed is None:
        numpy_seed = default_seed
    if random_seed is None:
        random_seed = default_seed
    if torch_seed is None:
        torch_seed = default_seed
    if ecole_seed is None:
        ecole_seed = default_seed

    np.random.seed(numpy_seed)

    random.seed(random_seed)

    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ecole.seed(ecole_seed)


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    """
    This utility function splits a tensor and pads each split to make them all the same size, then stacks them.
    """
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    output = torch.stack([F.pad(slice_, (0, max_pad_size - slice_.size(0)), 'constant', pad_value)
                          for slice_ in output], dim=0)
    return output


class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite` 
    observation function in a format understood by the pytorch geometric data handlers.
    """
    def __init__(self, 
                 observation=None,
                 candidates=None,
                 candidate_choice=None,
                 candidate_scores=None,
                 score=None):
        super().__init__()

        if observation is not None:
            self.row_features = torch.FloatTensor(observation.row_features)
            self.variable_features = torch.FloatTensor(observation.variable_features)
            self.edge_index = torch.LongTensor(observation.edge_features.indices.astype(np.int64))
            self.edge_attr = torch.FloatTensor(observation.edge_features.values).unsqueeze(1)
        if candidates is not None:
            self.candidates = torch.LongTensor(candidates)
            self.num_candidates = len(candidates)
        if candidate_choice is not None:
            self.candidate_choices = torch.LongTensor(candidate_choice)
        if candidate_scores is not None:
            self.candidate_scores = torch.FloatTensor(candidate_scores)
        if score is not None:
            self.score = torch.FloatTensor(score)

    def __inc__(self, key, value, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs 
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == 'edge_index':
            return torch.tensor([[self.row_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'candidates':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value)


class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files
        self.get_num_nodes = (lambda obs: obs.row_features.shape[0] + obs.variable_features.shape[0])

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        sample_observation, sample_action, sample_action_set, sample_scores = sample
        
        # We note on which variables we were allowed to branch, the scores as well as the choice 
        # taken by strong branching (relative to the candidates)
        candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
        try:
            candidate_scores = torch.FloatTensor([sample_scores[j] for j in candidates])
            score = []
        except (TypeError, IndexError):
            # only given one score and not in a list so not iterable
            score = torch.FloatTensor([sample_scores])
            candidate_scores = []
        candidate_choice = torch.where(candidates == sample_action)[0][0]

        graph = BipartiteNodeData(sample_observation, candidates, candidate_choice, candidate_scores, score)
        
        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = self.get_num_nodes(sample_observation)
        
        return graph
