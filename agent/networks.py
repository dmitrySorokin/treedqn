import torch
from torch import nn
import torch_geometric
import numpy as np
from ecole.observation import NodeBipartiteObs   

class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    def __init__(self):
        super().__init__('add')
        emb_size = 64
        
        self.feature_module_left = nn.Sequential(
           nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = nn.Sequential(
           nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = nn.Sequential(
           nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = nn.Sequential(
           nn.ReLU(inplace=True),
           nn.Linear(emb_size, emb_size)
        )

        # output_layers
        self.output_module = nn.Sequential(
           nn.Linear(2*emb_size, emb_size),
           nn.ReLU(inplace=True),
           nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]), 
                                node_features=(left_features, right_features), edge_features=edge_features)
        return self.output_module(torch.cat([output, right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(self.feature_module_left(node_features_i) 
                                           + self.feature_module_edge(edge_features) 
                                           + self.feature_module_right(node_features_j))
        return output


class GNNPolicy(nn.Module):
    def __init__(self, device, value_head=True):
        super().__init__()
        emb_size = 64
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 19
        self.device = device
        self.has_value_head = value_head

        # CONSTRAINT EMBEDDING
        self.cons_embedding = nn.Sequential(
           nn.Linear(cons_nfeats, emb_size),
           nn.ReLU(inplace=True),
           nn.Linear(emb_size, emb_size),
           nn.ReLU(inplace=True),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = nn.Sequential(
           nn.Linear(var_nfeats, emb_size),
           nn.ReLU(inplace=True),
           nn.Linear(emb_size, emb_size),
           nn.ReLU(inplace=True),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.output_module = nn.Sequential(
           nn.Linear(emb_size, emb_size),
           nn.ReLU(inplace=True),
           nn.Linear(emb_size, 1, bias=False),
        )

        if self.has_value_head:
            self.value = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(emb_size, 1),
            )

    def _unpack(self, obs):
        if not isinstance(obs, NodeBipartiteObs):
            return obs.row_features.to(self.device), \
                obs.edge_index.to(self.device), \
                obs.edge_attr.to(self.device),  \
                obs.variable_features.to(self.device)
        
        return torch.from_numpy(obs.row_features.astype(np.float32)).to(self.device), \
            torch.LongTensor(obs.edge_features.indices.astype(np.int16)).to(self.device), \
            torch.from_numpy(obs.edge_features.values.astype(np.float32)).view(-1, 1).to(self.device), \
            torch.from_numpy(obs.variable_features.astype(np.float32)).to(self.device)

    def forward(self, obs):
        constraint_features, edge_indices, edge_features, variable_features = self._unpack(obs)
    
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        
        constraint_features = self.cons_embedding(constraint_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)

        output = self.output_module(variable_features).squeeze(-1)
        if not self.has_value_head:
            return output
        
        value = self.value(variable_features).squeeze(-1)

        return value, output
