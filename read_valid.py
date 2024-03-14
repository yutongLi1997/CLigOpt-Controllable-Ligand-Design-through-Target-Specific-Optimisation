import json
import numpy as np
from torch_geometric.data import Data
import torch
from torch_geometric.data import InMemoryDataset

with open('data/molecules_zinc_valid.json', 'r') as f:
    data_valid = json.load(f)

def graph2tensor(data_dict):
    x_dim = len(data_dict['node_features_in'][0])
    n_active_nodes_in = len(data_dict['node_features_in'])
    chosen_bucket_size = len(data_dict['node_features_out'])
    x_in = data_dict['node_features_in']+ [[0 for _ in range(x_dim)] for __ in
                                              range(chosen_bucket_size - n_active_nodes_in)]
    x_out = data_dict['node_features_out']
    smiles_in = [*(data_dict['smiles_in'].strip().encode('ASCII'))]+[0]
    smiles_out = [*(data_dict['smiles_out'].strip().encode('ASCII'))]+[0]
    exit_points = data_dict['exit_points']
    abs_dist =  [float(x) for x in data_dict['abs_dist']]
    v_to_keep = data_dict['v_to_keep']
    
    edges_in0 = [data_dict['graph_in'][i][0] for i in range(len(data_dict['graph_in']))]
    edges_in1 = [data_dict['graph_in'][i][2] for i in range(len(data_dict['graph_in']))]
    edge_type_in = [data_dict['graph_in'][i][1] for i in range(len(data_dict['graph_in']))]
    edge_index_in = [edges_in0+edges_in1, edges_in1+edges_in0]
    edge_type_in += edge_type_in
    
    edges_out0 = [data_dict['graph_out'][i][0] for i in range(len(data_dict['graph_out']))]
    edges_out1 = [data_dict['graph_out'][i][2] for i in range(len(data_dict['graph_out']))]
    edge_type_out = [data_dict['graph_out'][i][1] for i in range(len(data_dict['graph_out']))]
#     edge_index_out = [edges_out0, edges_out1]
    edge_index_out = [edges_out0+edges_out1, edges_out1+edges_out0]
    edge_type_out += edge_type_out
    
    mask_in = np.zeros(chosen_bucket_size) 
    mask_in[v_to_keep] = 1
    graph_state_mask_in = torch.FloatTensor(mask_in)
    
    return Data(x=torch.tensor(x_in, dtype=torch.int),
                edge_index=torch.tensor(edge_index_in, dtype=torch.long),
                edge_type=torch.tensor(edge_type_in,dtype=torch.int),
                smiles_in = torch.CharTensor(smiles_in),
                x_out=torch.tensor(x_out, dtype=torch.int),
                edge_index_out=torch.tensor(edge_index_out, dtype=torch.long),
                edge_type_out=torch.tensor(edge_type_out,dtype=torch.int),
                smiles_out = torch.CharTensor(smiles_out),
                abs_dist = torch.FloatTensor(abs_dist),
                exit_points = torch.IntTensor(exit_points),
                v_to_keep = torch.IntTensor(v_to_keep),
                graph_state_mask_in = graph_state_mask_in
               )

class ZINCDataset(InMemoryDataset):
    def __init__(self, root,transform=None, pre_transform=None):
        super(ZINCDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['../data/zinc_processed_valid.dataset']

    def download(self):
        pass
    
    def process(self):
        
        data_list = []

        # process by session_id
        for data_dict in data_valid:
            data = graph2tensor(data_dict)
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])