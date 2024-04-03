# from torch_geometric.nn import GatedGraphConv
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool as gap
import torch
import torch.nn
from torch import Tensor
from torch.nn import Parameter as Param
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GATConv
import torch_geometric.data as pyg_data

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor

from torch_geometric.nn.inits import uniform, glorot

import numpy as np
import torch.nn.functional as F

# from utils import generate_std_normal

SMALL_NUMBER = 1e-7
LARGE_NUMBER= 1e10

def masked_edge_index(edge_index, edge_mask):
    return edge_index[:, edge_mask]

class OptGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(OptGCN, self).__init__()
        self.layer1 = GCNConv(in_dim, hidden_dim*4, bias = False)
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim * 4),
            torch.nn.LayerNorm(hidden_dim * 4),
            torch.nn.ReLU(inplace=True),
        )
        
        self.layer2 = GCNConv(hidden_dim * 8, hidden_dim * 4)
        self.norm2 = torch.nn.LayerNorm(hidden_dim * 4)
        self.layer3 = GCNConv(hidden_dim * 4, out_dim)
        self.norm3 = torch.nn.LayerNorm(out_dim)
    def forward(self, x, edge_index):
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.layer1(x, edge_index)
        x1 = F.relu(x1, inplace=True)
        f1 = self.fc1(x)
        x1f1 = torch.cat((x1, f1), 1)

        x2 = self.layer2(x1f1, edge_index)
        x2 = self.norm2(x2)
        x2 = F.relu(x2, inplace=True)

        x3 = self.layer3(x2, edge_index)
        x3 = self.norm3(x3)
        # x3 = F.relu(x3, inplace=True)

        # readout = pyg_data.batch(x3, batch)
        return x3

class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.egnn = OptGCN(input_size, hidden_size, output_size)

    def forward(self, x, edge_index):
        x = self.egnn(x, edge_index)
        return x
    
def pad_annotations(annotations, num_hidden = 32, num_nodes = 14, device = None):
    annotations_np = np.pad(annotations.cpu(), pad_width=[[0, 0], [0, num_hidden - num_nodes]],
                       mode='constant')
    return torch.from_numpy(annotations_np).float().to(device)

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fun = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size*8, bias=False),
            torch.nn.LayerNorm(hidden_size*8),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(hidden_size*8, hidden_size*2, bias=False),
            torch.nn.LayerNorm(hidden_size*2),
            torch.nn.Dropout(0.1),
            #nn.ReLU(inplace=True),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(hidden_size*2, output_size, bias=False)
        )

    def forward(self, x):
        x = self.fun(x)
        return x
    
class Attention(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.lin_attention_c = torch.nn.Linear(hidden_size, hidden_size)
        self.lin_attention_y = torch.nn.Linear(hidden_size, hidden_size)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, num_graphs, nv, z, mask):
        
        atten_mask_c = torch.tile(mask.reshape((num_graphs,nv,-1)).unsqueeze(2),[1,1,nv,1])* LARGE_NUMBER - LARGE_NUMBER
        atten_mask_yi = torch.tile(mask.reshape((num_graphs,nv,-1)).unsqueeze(1),[1,nv,1,1]) * LARGE_NUMBER - LARGE_NUMBER
        atten_mask = atten_mask_c + atten_mask_yi

        z = z.reshape([num_graphs,nv, self.hidden_size])
        # Combine fragments node embeddings with full molecule embedding
        # Attention mechanism over in_mol encodings to determine combination with z_sampled
        atten_c = torch.tile(z.reshape((-1,nv,self.hidden_size)).unsqueeze(2),[1,1,nv,1])
        atten_yi = torch.tile(z.reshape((-1,nv,self.hidden_size)).unsqueeze(1),[1,nv,1,1])# [b, v, v, h]
        atten_c = self.lin_attention_c(atten_c) # [b, v, v, h]
        atten_yi = self.lin_attention_y(atten_yi) # [b, v, v, h]
        atten_mi = self.sigmoid(torch.add(atten_c, atten_yi) + atten_mask) # [b,v,v,h]
        atten_mi = torch.sum(atten_mi, 2) / torch.tile(torch.unsqueeze(torch.sum(mask.reshape((num_graphs,nv,-1)),1),1),[1,nv,1]) # [b,v,h]

        atten_mi = atten_mi.reshape([-1, self.hidden_size])
        
        return atten_mi


class Decoder(torch.nn.Module):
    def __init__(self, hidden_size = 65, num_layers = 1, maximum_distance = 50):
        super(Decoder, self).__init__()
        # decoder
        self.hidden_size = hidden_size
        self.enc_dec = OptGCN(hidden_size, hidden_size, hidden_size)
        self.emb_distance = torch.nn.Embedding(maximum_distance, hidden_size)
        self.emb_overlapped_edge = torch.nn.Embedding(2, hidden_size)
        self.fc_edge_logits = MLP(6*hidden_size+3, 6*hidden_size+3, 1)
        self.stop_node = torch.nn.Parameter(torch.tensor(torch.nn.init.xavier_normal(torch.empty((1, hidden_size)))), requires_grad=True)
        self.fc_edge_type0 = MLP(6*hidden_size+3, 6*hidden_size+3, 1)
        self.fc_edge_type1 = MLP(6*hidden_size+3, 6*hidden_size+3, 1)
        self.fc_edge_type2 = MLP(6*hidden_size+3, 6*hidden_size+3, 1)
        self.fc_edge_type3 = MLP(6*hidden_size+3, 6*hidden_size+3, 1)
        self.ac = torch.nn.Softmax()
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, z, data_at_step, dist, idx, num_grpahs):
        
        device = z.device
        data_at_step = data_at_step.to(device)
        edge_index, edge_type, node_sequence = data_at_step.edge_index, data_at_step.edge_type, data_at_step.node_sequence
        edge_masks = data_at_step.edge_masks * LARGE_NUMBER - LARGE_NUMBER
        edge_type_mask = data_at_step.edge_type_mask * LARGE_NUMBER - LARGE_NUMBER
        
        
        node_sequence = node_sequence.unsqueeze(1) # [b*v, 1]

        total_v = node_sequence.shape[0]
        nv = int(total_v/num_grpahs)
        z = torch.cat((z,node_sequence), dim = 1) # [b*v, 2h+1]

        new_z = self.enc_dec(z,edge_index) # encoder on z, # [b*v, 2h+1]

        # edge features
        # Take out the node in focus
        node_in_focus = torch.sum((node_sequence*new_z).view(num_grpahs,-1,self.hidden_size), 1) # [b, 2h+1]
        # h_u = new_z
        
        # edge features
        edge_repr = torch.cat((node_in_focus.unsqueeze(1).expand(num_grpahs,nv,self.hidden_size).reshape(-1,self.hidden_size), new_z),1) # [b*v, 2*(h+h+1)]
        local_graph_repr_before_expansion = torch.sum(new_z.reshape(num_grpahs,nv,self.hidden_size),1) # [b, 2h+1]
        local_graph_repr = local_graph_repr_before_expansion.unsqueeze(1).expand(num_grpahs,nv,self.hidden_size).reshape(-1,self.hidden_size) # [b*v, 2h+1]
        global_graph_repr_before_expansion = torch.sum(z.reshape(num_grpahs,nv,self.hidden_size),1) # [b, 2h+1]
        global_graph_repr = global_graph_repr_before_expansion.unsqueeze(1).expand(num_grpahs,nv,65).reshape(-1,self.hidden_size) # [b*v,2h+1]
        distance_repr = self.emb_distance(data_at_step.distance) # [b*v,2h+1]
        overlapped_edge_repr = self.emb_overlapped_edge(data_at_step.overlapped_edge_features) # [b*v,2h+1]
        combined_edge_repr = torch.cat((edge_repr, local_graph_repr, global_graph_repr, distance_repr, overlapped_edge_repr), 1) # [b*v,6*(2h+1)]
        
        # Add structural info (dist, ang) and iteration number
        it_num = torch.tile(torch.FloatTensor([idx]).unsqueeze(0).to(device), [total_v, 1]) # [v,1]
        pos_info = torch.cat([dist.reshape(num_grpahs,-1).unsqueeze(1).expand(num_grpahs,nv,2).reshape(-1,2), it_num], axis=1)
        combined_edge_repr = torch.cat([combined_edge_repr, pos_info], axis=1) # [b*v, 6(2h+1)+2+1]
        
        # stop node features
        distance_to_stop_node = self.emb_distance(torch.tensor([0], device=z.device)).expand(num_grpahs,self.hidden_size)
        overlap_edge_stop_node = self.emb_overlapped_edge(torch.tensor([0], device=z.device)).expand(num_grpahs,self.hidden_size)
        combined_stop_node_repr = torch.cat((node_in_focus, self.stop_node.expand(num_grpahs, self.hidden_size), local_graph_repr_before_expansion, 
                                     global_graph_repr_before_expansion, distance_to_stop_node, overlap_edge_stop_node, torch.mean(pos_info.reshape(num_grpahs,-1,3), 1)), 1) # [1, 6*(2h+1)+3]
        
        # edge logits
        edge_logits = self.fc_edge_logits(combined_edge_repr.reshape(num_grpahs,-1,6*self.hidden_size+3)) # [b, v, 1]
        edge_logits = edge_logits.squeeze()+edge_masks.reshape(num_grpahs,-1) # [b, v] 
        stop_logits = self.fc_edge_logits(combined_stop_node_repr) # [b,1]
        edge_logits = torch.cat([edge_logits, stop_logits], axis=1) # [b, v + 1]
        
#         edge_probs = torch.log(self.ac(edge_logits)+ SMALL_NUMBER) # [v+1]
        edge_probs = self.ac(edge_logits) # [v+1]
        # edge_probs = self.sigmoid(edge_logits)
        #edge type logits
        edge_type_logit0 = self.fc_edge_type0(combined_edge_repr.reshape(num_grpahs, -1, 6*self.hidden_size+3)) # [b, v, 1]
        edge_type_logit1 = self.fc_edge_type1(combined_edge_repr.reshape(num_grpahs, -1, 6*self.hidden_size+3))
        edge_type_logit2 = self.fc_edge_type2(combined_edge_repr.reshape(num_grpahs, -1, 6*self.hidden_size+3))
        edge_type_logit3 = self.fc_edge_type3(combined_edge_repr.reshape(num_grpahs, -1, 6*self.hidden_size+3))

        edge_type_logits = torch.cat((edge_type_logit0,edge_type_logit1,edge_type_logit2, edge_type_logit3),2) # [b, v,4]
        edge_type_logits =  edge_type_logits.permute(0,2,1)+edge_type_mask.reshape(num_grpahs,4,nv)
        
#         edge_type_probs = torch.log(self.ac(edge_type_logits)+ SMALL_NUMBER) # [4, v]
        edge_type_probs = F.softmax(edge_type_logits, dim = 1) # [4, v]
        # edge_type_probs = F.softmax(edge_type_logits, dim = 1)
        
        return edge_probs, edge_type_probs
#         return edge_logits, edge_type_logits
def generate_std_normal(a1, a2):
    return np.random.normal(0, 1, [a1, a2])

class OptLinker(torch.nn.Module):
    def __init__(self, hidden_size = 32, num_layers = 1, maximum_distance = 50, encoding_size = 4):
        super(OptLinker, self).__init__()
        
        self.hidden_size = hidden_size
        self.out_dim = encoding_size
        self.embed = torch.nn.Embedding(14, hidden_size)
        # self.embed_out = torch.nn.Embedding(14, hidden_size)
        self.encoder = Encoder(hidden_size, hidden_size, hidden_size)
        self.lin_mu = torch.nn.Linear(hidden_size, hidden_size)
        self.lin_logvariance = torch.nn.Linear(hidden_size, hidden_size)
        self.lin_mu_out = torch.nn.Linear(hidden_size, encoding_size) 
        self.lin_logvariance_out = torch.nn.Linear(hidden_size, encoding_size)
        self.atten = Attention(hidden_size)
        self.lin_mean_combine_weights_in = torch.nn.Linear(encoding_size, hidden_size)
        # self.embed_dec = torch.nn.Linear(hidden_size, hidden_size)
        
        self.decoder = Decoder(hidden_size*2+1, num_layers, maximum_distance)

        self.atten_node = Attention(hidden_size+3)
        self.lin_mean_combine_weights_in_node = torch.nn.Linear(hidden_size+3, hidden_size+3)

        self.lin_node_symbol = torch.nn.Linear(hidden_size+3,14)
        
        self.softmax = torch.nn.Softmax()
    
    def forward(self, data, incremental_results):
        num_graphs = data.num_graphs
        
        x_in, edge_index_in, edge_type_in, x_out, edge_index_out, edge_type_out, graph_state_mask_in, abs_dist= \
        data.x, data.edge_index, data.edge_type.float(), data.x_out, data.edge_index_out, data.edge_type_out.float(), data.graph_state_mask_in, data.abs_dist
        device = x_in.device
        
        nv = int(x_out.size()[0]/num_graphs)
        graph_state_mask_in1 = graph_state_mask_in.unsqueeze(1)
#------------------------------------ Encoder ------------------------------------
        x_padded_in = pad_annotations(x_in,device = device) # [b*v, h]
#         x_embed_in = self.embed(x_padded_in) # [b*v_in,h]
        x_embed_in = self.embed(torch.argmax(x_padded_in, axis = 1)) * graph_state_mask_in1
        x_encoded_in = self.encoder(x_embed_in, edge_index_in) # [b*v_in,h]

        x_padded_out = pad_annotations(x_out,device = device) # [b*v, h]
        x_embed_out = self.embed(torch.argmax(x_padded_out, axis = 1)) # [b*v_in,h]
        x_encoded_out = self.encoder(x_embed_out, edge_index_out) # [b*v_in,h]
        
#-------------------------------------Compute mean & variance-----------------------

        mean = self.lin_mu(x_encoded_in) # [b*v, h]
        logvariance = self.lin_logvariance(x_encoded_in) # [b*v, h]
        sigma = torch.exp(logvariance)

        avg_last_h_out = gap(x_encoded_out, data.batch) # [b,h]
        mean_out = self.lin_mu_out(avg_last_h_out) #[b,4]
        logvariance_out = self.lin_logvariance_out(avg_last_h_out) # [b,4]
        mean_out_ex = torch.tile(mean_out.unsqueeze(1), [1, nv, 1]).reshape((-1, self.out_dim)) # [b*v,4]
        logvariance_out_ex = torch.tile(logvariance_out.unsqueeze(1), [1, nv, 1]).reshape((-1, self.out_dim))# [b*v,4]
        sigma_out = torch.exp(logvariance_out_ex)# [b*v,4]
        
#---------------------------------------- Distribution ------------------------------------

#         distrib = torch.distributions.Normal(loc=mean, scale=sigma)
# #       sample o from distribution
#         # o = distrib.sample()

#         distrib_out = torch.distributions.Normal(loc=mean_out_ex, scale=sigma_out)
# #       sample o from distribution
#         o_out = distrib_out.sample() # [b,v,4]
        
        z_prior = torch.distributions.Normal(0,1).sample((x_out.size()[0], self.out_dim)).to(device) #[b*v,o]
        z_prior_in = torch.distributions.Normal(0,1).sample((x_in.size()[0], self.hidden_size)).to(device) #[b*v,h]

        z_sampled = mean_out_ex + sigma_out*z_prior


        loss_kl_in = 1+logvariance-torch.square(mean)-torch.exp(logvariance)
        loss_kl_in = loss_kl_in.sum(-1)*graph_state_mask_in

        loss_kl_out = 1+logvariance_out_ex-torch.square(mean_out_ex)-torch.exp(logvariance_out_ex)
        loss_kl_out = loss_kl_out.sum(-1)

        loss_kl = -0.5*loss_kl_in.reshape((data.num_graphs, -1)).sum(-1).mean() -0.5* loss_kl_out.reshape((data.num_graphs, -1)).sum(-1).mean()
        
#-------------------------------------- Attention --------------------------------------

        
        graph_state_mask_out1 = Tensor(np.ones(graph_state_mask_in1.shape)).to(device)

        mean = mean*graph_state_mask_in1 # [b*v, h]
        # mean = mean.reshape([num_graphs,nv,self.hidden_size])# [b,v, h]
        inverted_mask = torch.ones(graph_state_mask_in1.shape).to(device) - graph_state_mask_in1 # [b*v,1]
        # inverted_mask = inverted_mask.reshape([num_graphs, nv, -1]) # [b,v,1]

        update_vals = z_prior_in*inverted_mask # [b*v,h]
        mean = torch.add(mean, update_vals) # [b*v,h]
        atten_mi = self.atten(num_graphs, nv, mean, graph_state_mask_out1)

        # o_out = o_out.reshape([-1, self.out_dim])
        z_sampled = self.lin_mean_combine_weights_in(z_sampled) # [b*v, h]
        mean_sampled = mean * graph_state_mask_out1 + atten_mi * z_sampled # [b*v,h]
        # mean_sampled = mean_sampled.reshape([-1, self.hidden_size]) # [b&v]

#------------------------------------------ Sample -------------------------------------------
#         lantent_node_state = self.embed_out(mean_sampled) # [b*v,h]
        lantent_node_state = x_embed_out
        
        z = torch.cat((mean_sampled, lantent_node_state), -1) # [b*v, 2h] # initial representation for decoder # filtered_z_sampled
        # z = torch.cat((mean_sampled, lantent_node_state), -1) 

        
#--------------------------------------------- Decoder --------------------------------------

        edge_probs = []
        edge_type_probs = []

        for i in range(len(incremental_results)):
            data_at_step = next(iter(incremental_results[i]))
            edge_probs_at_step, edge_type_probs_at_step = self.decoder(z, data_at_step, abs_dist, i,num_graphs)
            edge_probs.append(edge_probs_at_step)
            edge_type_probs.append(edge_type_probs_at_step)
            
#--------------------------------------------Node Symbol---------------------------------------
        
        atoms = (graph_state_mask_in.reshape(num_graphs,-1) ==0).nonzero(as_tuple=True)[0]
        num_atoms = torch.bincount(atoms) # [b]
        num_atoms = num_atoms.unsqueeze(1).expand(num_graphs,nv).unsqueeze(2) #[b,v,1]
        dist = abs_dist.reshape(num_graphs,-1).unsqueeze(1).expand(num_graphs,nv,2) #[b,v,2]
        
        pos_info = torch.cat([num_atoms, dist], axis=2).reshape(-1,3) # [b*v, 3]
        
        # z_sampled = torch.cat([lantent_node_state, pos_info], axis = 1) # [b*v, h+3]
        z_sampled = torch.cat([mean_sampled, pos_info], axis = 1) # [b*v, h+3]

# --------------------------------------- Attention ------------------------------------
        node_mask = graph_state_mask_out1 - graph_state_mask_in1
        atten_mi_node = self.atten_node(num_graphs, nv, z_sampled, node_mask)
        
        z_sampled_out = self.lin_mean_combine_weights_in_node(z_sampled)
        z_sampled = z_sampled*graph_state_mask_in1 + atten_mi_node*z_sampled_out # [b*v,h+3]

        node_logits = self.lin_node_symbol(z_sampled) # [b*v, 14]

        node_probs = self.softmax(node_logits) # [b*v, 14]
    
        return loss_kl, edge_probs, edge_type_probs, node_probs