# import json
from torch_geometric.data import Data
import torch
from torch_geometric.loader import DataLoader
# from torch_geometric.data import InMemoryDataset
from read_valid import *
from incremental_results import *
from torch import Tensor


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolDrawing, DrawingOptions
from rdkit.Chem import MolStandardize

import example_utils
import frag_utils
from data.prepare_data import read_file, preprocess
import re
import copy
from OptLinker_vanilla_GCN import *



def sample_node_symbol1(all_node_symbol_prob, all_lengths, dataset):
#     for graph_idx, graph_prob in enumerate(all_node_symbol_prob):
    node_symbol=[]
    for node_idx in range(len(all_node_symbol_prob)):
        symbol=np.random.choice(np.arange(len(dataset_info(dataset)['atom_types'])), p=all_node_symbol_prob[node_idx])
        node_symbol.append(symbol)
    return node_symbol

def generate_std_normal(a1, a2):
    return np.random.normal(0, 1, [a1, a2])

def node_symbol_one_hot(sampled_node_symbol, real_n_vertices):
    one_hot_representations=[]
    for idx in range(real_n_vertices):
        representation = [0] * 14
        atom_type=sampled_node_symbol[idx]
        representation[atom_type]=1
        one_hot_representations.append(representation)
    return one_hot_representations

class DeLinker_Generator():
    def __init__(self, model, model_dti, device = None):
        self.model = model
        self.model_dti = model_dti
        self.device = device
        
    def gradient_ascent(self, random_normal_states, derivative_z_sampled, prior_learning_rate = 0.05):        
        return random_normal_states + prior_learning_rate * derivative_z_sampled
    
    def generate_new_graph(self, mol_tensor, mol_dict, seq, count,optimization_step=1):
#         incremental_result = get_incremental_results_in_train(data)
#         x, edge_index, edge_types = data.x, data.edge_index, data.edge_type
        num_vertices = mol_tensor.x.shape[0]

        # all generated similes
        generated_all_similes=[]

#         random_normal_states = Tensor(generate_std_normal(num_vertices, 100))
        random_normal_states = Tensor(generate_std_normal(num_vertices, self.model.out_dim))
        random_normal_states_in = Tensor(generate_std_normal(num_vertices, self.model.hidden_size))
        random_normal_states.requires_grad = True
        best_mol = self.optimization_over_prior(random_normal_states, random_normal_states_in, num_vertices, generated_all_similes, count, mol_tensor, mol_dict, seq, optimization_step)
        return best_mol
        
    def optimization_over_prior(self, random_normal_states, random_normal_states_in, num_vertices, generated_all_similes, count, mol_tensor, mol_dict, seq, optimization_step = 0):
        # record how many optimization steps are taken
        step=0
        # generate a new molecule
        best_mol = self.generate_graph_with_state(random_normal_states, random_normal_states_in, num_vertices, generated_all_similes, step, count, mol_tensor, mol_dict)
        for _ in range(optimization_step):
            z_normalized = torch.nn.functional.normalize(random_normal_states)
#             flattened_z_sampled = torch.reshape(random_normal_states, [1, -1])
#             l2_loss = 0.01* torch.sum(flattened_z_sampled * flattened_z_sampled, axis=1) /2
#             qed_pred = model.gated_regression(z_normalized)
            qed_pred = self.model_dti(random_normal_states, seq)
#             dYdX = torch.autograd.grad(torch.reshape(qed_pred, [1]) - l2_loss , random_normal_states, allow_unused=True)
            dYdX = torch.autograd.grad(torch.reshape(qed_pred, [1]), random_normal_states, allow_unused=True)

            # update the states
            random_normal_states=self.gradient_ascent(random_normal_states, dYdX[0])
            # generate a new molecule
            step+=1
            self.generate_graph_with_state(random_normal_states, num_vertices, generated_all_similes, step, count, mol_tensor, mol_dict)
        return best_mol
    
    def generate_graph_with_state(self,random_normal_states, random_normal_states_in, num_vertices, generated_all_similes, step, count, mol_tensor,mol_dict,
                              num_different_starting = 6, dataset = 'zinc', number_of_generation = 30000):
        # Get predicted node probs
        #------------Attention-------------
        x_in, edge_index_in, edge_type_in, x_out, edge_index_out, edge_type_out, graph_state_mask_in, abs_dist= \
        mol_tensor.x, mol_tensor.edge_index, mol_tensor.edge_type.float(), mol_tensor.x_out, mol_tensor.edge_index_out, mol_tensor.edge_type_out.float(), mol_tensor.graph_state_mask_in, mol_tensor.abs_dist
        device = x_in.device

        x_padded_in = pad_annotations(x_in,device = device) # [b*v, h]
        x_embed_in = self.model.embed_in(torch.argmax(x_padded_in, axis = 1)) # [b*v_in,h]
        x_encoded_in = self.model.encoder(x_embed_in, edge_index_in) # [b*v_in,h]
        # x_encoded_in = self.model.encoder(x_embed_in, edge_index_in, edge_type_in) # [b*v_in,h]

        x_padded_out = pad_annotations(x_out,device = device) # [b*v, h]
        x_embed_out = self.model.embed_out(torch.argmax(x_padded_out, axis = 1)) # [b*v_in,h]
        x_encoded_out = self.model.encoder(x_embed_out, edge_index_out) # [b*v_in,h]
        # x_encoded_out = self.model.encoder(x_embed_out, edge_index_out, edge_type_out) # [b*v_in,h]

        mean = self.model.lin_mu(x_encoded_in) # [b*v, h]

        graph_state_mask_in1 = graph_state_mask_in.unsqueeze(1)
        mean = mean*graph_state_mask_in1 # [b*v, h]
        inverted_mask = torch.ones(graph_state_mask_in1.shape).to(device) - graph_state_mask_in1 # [b*v,1]

        update_vals = random_normal_states_in*inverted_mask # [b*v,h]
        mean = torch.add(mean, update_vals) # [b*v,h]
        atten_mi = self.model.atten(1, num_vertices, mean)

        random_normal_states = self.model.lin_mean_combine_weights_in(random_normal_states) # [b*v, h]
        random_normal_states = mean + atten_mi * random_normal_states # [b*v,h]
        
        # lantent_node_state = self.model.embed_out(random_normal_states)
        lantent_node_state = x_embed_out
        abs_dist = torch.tile(mol_tensor.abs_dist.unsqueeze(0),[num_vertices,1]) # [v,2]
        num_atoms = len((mol_tensor.graph_state_mask_in == 0).nonzero(as_tuple=True)[0])
        num_atoms = torch.tile(torch.FloatTensor([num_atoms]).unsqueeze(0).to(self.device), [num_vertices, 1]) # [v,1]
        pos_info = torch.cat([abs_dist, num_atoms], axis=1) # [v, 3]

        z_sampled = torch.cat([lantent_node_state, pos_info], axis = 1) # [b*v, h+3]
        # z_sampled = torch.cat([random_normal_states, pos_info], axis = 1) # [b*v, h+3]
    
        z = torch.cat((random_normal_states, lantent_node_state),-1)
        #-------node symbol attention -----

        atten_mi_node = self.model.atten_node(1, num_vertices, z_sampled)
        
        z_sampled_out = self.model.lin_mean_combine_weights_in_node(z_sampled)
        z_sampled = z_sampled*graph_state_mask_in1 + atten_mi_node*z_sampled_out # [b*v,h+3]

        node_logits = self.model.lin_node_symbol(z_sampled) # [b*v, 14]

        # node_probs = self.softmax(node_logits) # [b*v, 14]
        
        
        # node_logits = self.model.lin_node_symbol(z_sampled.squeeze(0)) # [v, 14]
        predicted_node_symbol_prob = self.model.softmax(node_logits)
        print(predicted_node_symbol_prob)
        
        # Sample node symbols
        sampled_node_symbol=sample_node_symbol1(predicted_node_symbol_prob.cpu().detach().numpy(), num_vertices, dataset)
        # print(sampled_node_symbol)
        # Maximum valences for each node
        sampled_node_keep = mol_tensor.v_to_keep # [v]
        for node, keep in enumerate(sampled_node_keep): 
            sampled_node_symbol[node] = np.argmax(mol_tensor.x[node].numpy())
        valences=get_initial_valence(sampled_node_symbol, dataset) # [v]
        
        # randomly pick the starting point or use zero 
        starting_point=random.sample(range(num_vertices-10, num_vertices), min(num_different_starting, num_vertices-1))
        # print(starting_point)
#         starting_point = [30,31,32,33,34,35,36,37,38]
    
        all_mol=[]
        for idx in starting_point: 
#             new_mol, total_log_prob=self.search_and_generate_molecule(idx, copy.deepcopy(valences),
#                                                 sampled_node_symbol, num_vertices,
#                                                 random_normal_states, z)
            new_mol, total_log_prob=self.search_and_generate_molecule(idx, copy.deepcopy(valences),
                                                                      sampled_node_symbol, mol_tensor, mol_dict, num_vertices, z)
            # record the molecule with largest number of shapes
            if dataset=='qm9' and new_mol is not None:
                all_mol.append((np.sum(shape_count(dataset, True, 
                                [Chem.MolToSmiles(new_mol)])[1]), total_log_prob, new_mol))
            # record the molecule with largest number of pentagon and hexagonal for zinc and cep
            elif dataset=='zinc' and new_mol is not None:
                counts=shape_count(dataset, True,[Chem.MolToSmiles(new_mol)])
                all_mol.append((0.5 * counts[1][2]+ counts[1][3], total_log_prob, new_mol))
                # all_mol.extend((0.5 * counts[1][2]+ counts[1][3], total_log_prob, new_mol))
            elif dataset=='cep' and new_mol is not None:
                all_mol.append((np.sum(shape_count(dataset, True,
                                [Chem.MolToSmiles(new_mol)])[1][2:]), total_log_prob, new_mol))

        # select one out
        try:
            best_mol = select_best(all_mol)
        except:
            return
        # nothing generated
        if best_mol is None:
            return
        # visualize it 
        make_dir('visualization_%s' % dataset)
        visualize_mol('visualization_%s/%d_%d.png' % (dataset, count, step), best_mol)
#         for (i,m) in enumerate(all_mol):
#             visualize_mol('visualization_%s/%d_%d.png' % (dataset, i, step), m)
        # record the best molecule
        smi = Chem.MolToSmiles(best_mol)
        generated_all_similes.append(Chem.MolToSmiles(best_mol))
        with open('generated_smiles.txt', 'w') as f:
            for smi in generated_all_similes:
                f.write(f"{smi}\n")
        # dump('generated_smiles_%s' % (dataset), generated_all_similes)
        print("Real QED value")
        print(QED.qed(best_mol))
        if len(generated_all_similes) >= number_of_generation:
            print("generation done")
            exit(0)
        print(smi)
        return best_mol
#         return smi
            
    def search_and_generate_molecule(self, initial_idx, valences, sampled_node_symbol, mol_tensor, mol_dict, real_n_vertices, 
                                 z):
        # New molecule
        new_mol = Chem.MolFromSmiles('')
        new_mol = Chem.rdchem.RWMol(new_mol)
        # Add atoms
        add_atoms(new_mol, sampled_node_symbol, 'zinc')
        # Breadth first search over the molecule
        queue=deque([initial_idx])
        # queue = deque([])
        # color 0: have not found 1: in the queue 2: searched already
        color = [0] * real_n_vertices
        color[initial_idx] = 1
        # Empty adj list at the beginning
        incre_adj_list=defaultdict(list)
        # record the log probabilities at each step

        adj_mat_in = graph_to_adj_mat(mol_dict['graph_in'], real_n_vertices, 4)
        sampled_node_keep = node_keep_to_dense(mol_tensor.v_to_keep, real_n_vertices)
        count_bonds = 0
        # Add edges between vertices to keep 
        for node, keep in enumerate(sampled_node_keep[0:real_n_vertices]): 
            if keep == 1:
                for neighbor, keep_n in enumerate(sampled_node_keep[0:real_n_vertices]): 
                    if keep_n == 1 and neighbor > node:
                        for bond in range(4):
                            if adj_mat_in[bond][node][neighbor] == 1:
                                incre_adj_list[node].append((neighbor, bond))
                                incre_adj_list[neighbor].append((node, bond))
                                valences[node] -= (bond+1)
                                valences[neighbor] -= (bond+1)
                                #add the bond
                                new_mol.AddBond(int(node), int(neighbor), number_to_bond[bond])
                                count_bonds += 1


        # Add exit nodes to queue and update colours of fragment nodes
        for v, keep in enumerate(sampled_node_keep[0:real_n_vertices]):
            if keep == 1:
                if v in mol_tensor.exit_points:
                    queue.append(v)
                    color[v]=1
                else:
                    # Mask out nodes that aren't exit vectors
                    valences[v] = 0
                    color[v] = 2

        total_log_prob=0

        nidx = 0
        # queue.append(mol_tensor.exit_points.detach().numpy()[0])
        try_another_time = 0
        double_bond_check = []
        while len(queue) > 0:
            
            node_in_focus = queue.popleft()
#             if valences[node_in_focus] == 0:
#                 continue
            while True:
                # Prepare data for one iteration based on the graph state
                edge_type_mask_sparse, edge_mask_sparse = generate_mask(valences, incre_adj_list, color, 
                                                                        real_n_vertices, node_in_focus, False, new_mol)
                edge_type_mask = edge_type_masks_to_dense([edge_type_mask_sparse], real_n_vertices, 4) # [e, v]
                edge_mask = edge_masks_to_dense([edge_mask_sparse],real_n_vertices) # [v]
        #                 print(edge_mask)
                node_sequence = node_sequence_to_dense([node_in_focus], real_n_vertices) # [v]
                distance_to_others_sparse = bfs_distance(node_in_focus, incre_adj_list)
                distance_to_others = distance_to_others_dense([distance_to_others_sparse],real_n_vertices) # [v]
                overlapped_edge_sparse = get_overlapped_edge_feature(edge_mask_sparse, color, new_mol)

                overlapped_edge_dense = overlapped_edge_features_to_dense([overlapped_edge_sparse],real_n_vertices) # [v]
                incre_adj_mat = incre_adj_mat_to_dense([incre_adj_list], 
                    4, real_n_vertices) # [e, v, v]
                sampled_node_symbol_one_hot = node_symbol_one_hot(sampled_node_symbol, real_n_vertices)

                d_list = self.get_dynamic_feed_dict(sampled_node_symbol_one_hot, incre_adj_mat, distance_to_others, overlapped_edge_dense, node_sequence, 
                                          edge_type_mask, edge_mask)
                # predict edge_probs & edge_type_probs
                # dist = torch.tile(mol_tensor.abs_dist.unsqueeze(0),[real_n_vertices,1]) # [v,2]
                dist = mol_tensor.abs_dist
                edge_probs, edge_type_probs = self.model.decoder(z, d_list[0], dist,nidx, 1)
                nidx+=1
                # print(edge_type_probs)

                # print(np.argmax(edge_probs.detach().numpy()))
                neighbor=np.random.choice(np.arange(real_n_vertices+1), p=edge_probs.detach().numpy()[0])
                # print(neighbor)
                # neighbor=np.argmax(edge_probs.detach().numpy())
                # print(str(node_in_focus)+'pair with' + str(neighbor))
                # print(edge_probs.shape)
                # update log prob
                total_log_prob+=np.log(edge_probs.detach().numpy()[0,neighbor]+SMALL_NUMBER)
                # stop it if stop node is picked
                if neighbor == real_n_vertices:
                    break
                               
                # or choose an edge type
                # print(edge_type_probs.detach().numpy()[0, :, neighbor])
                bond = np.random.choice(np.arange(4), p = edge_type_probs.detach().numpy()[0, :, neighbor])
                # bond=np.argmax(edge_type_probs.detach().numpy()[0, :, neighbor])

                # bond=np.argmax(tmp1.detach().numpy()[:, neighbor])
                # update log prob
                total_log_prob+=np.log(edge_type_probs.detach().numpy()[0, :, neighbor][bond]+SMALL_NUMBER)
                #update valences
#                 if valences[node_in_focus] < bond+1 or valences[neighbor] < bond+1 or neighbor == node_in_focus or color[neighbor] == 2:
#                     continue
#                 if not new_mol.GetBondBetweenAtoms(int(node_in_focus),int(neighbor)) == None:
#                     continue
                if valences[node_in_focus] < bond+1 or valences[neighbor] < bond+1:
                    # if node_in_focus in mol_tensor.exit_points and valences[node_in_focus]!=0:
                    #     continue
                    # else:
                    break
                # if neighbor == node_in_focus or color[neighbor] == 2:
                if neighbor == node_in_focus:
                    continue
                if not new_mol.GetBondBetweenAtoms(int(node_in_focus),int(neighbor)) == None:
                    continue
                # print(double_bond_check)
                if bond == 2:
                    bond = 1
                if bond == 1:
                    # print(node_in_focus)
                    # print(neighbor)
                    if node_in_focus in double_bond_check or neighbor in double_bond_check: 
                        bond = 0
#                 print(node_in_focus)
#                 print(neighbor)
                valences[node_in_focus] -= (bond+1)
                valences[neighbor] -= (bond+1)
                # valences[node_in_focus] -= bond
                # valences[neighbor] -= bond
                # print('sucess')

                #add the bond
                new_mol.AddBond(int(node_in_focus), int(neighbor), number_to_bond[bond])
                if bond == 1:
                    # print(node_in_focus)
                    double_bond_check.append(node_in_focus)
                    # print(neighbor)
                    double_bond_check.append(neighbor)
                # print(double_bond_check)
                # add the edge to increment adj list
    #             print(incre_adj_list[node_in_focus])
                incre_adj_list[node_in_focus].append((neighbor, bond))
                incre_adj_list[neighbor].append((node_in_focus, bond))
                # Explore neighbor nodes
                if color[neighbor]==0:
                    queue.append(neighbor) 
                    color[neighbor]=1
            color[node_in_focus]=2    # explored
            # if len(queue) == 0:
            #     remove_extra_nodes(new_mol)
            #     if len(new_mol.GetAtoms()) < mol_tensor.v_to_keep.detach().numpy().shape[0]:
            #         queue.append(mol_tensor.exit_points.detach().numpy()[0])
            #         queue.append(mol_tensor.exit_points.detach().numpy()[1])
            # Remove unconnected node     
#         print(Chem.MolToSmiles(new_mol))
        remove_extra_nodes(new_mol)
        if len(new_mol.GetAtoms()) < mol_tensor.v_to_keep.detach().numpy().shape[0]:
            return None, total_log_prob
#         print(Chem.MolToSmiles(new_mol))
        # new_mol=Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))
        
        return new_mol, total_log_prob
    def get_dynamic_feed_dict(self, sampled_node_symbol_one_hot, incre_adj_mat, distance, overlapped_edge_features, node_sequence, 
                          edge_type_mask, edge_mask):
#     if edge_index is None:
#         latent_node_symbol = np.zeros((1, 14))
#         edge_index = np.zeros((2, 2))
#         edge_types = np.zeros(2)
#         distance = np.zeros(num_vertices)
#         edge_type_mask = np.zeros((4, num_vertices))
# #         edge_type_labels = np.zeros((4, num_vertices))
#         edge_mask = np.zeros(num_vertices)
# #         edge_labels = np.zeros(num_vertices)
#         overlapped_edge_features = np.zeros(num_vertices)
#         node_sequence = np.zeros(num_vertices)
        
        
        device_kwargs = {"device":self.device} if self.device is not None else {}
        data_list = []

        for i in range(len(distance)):
            atoms_array = np.array(np.where(np.triu(incre_adj_mat[i], k = 1) == 1))
            # sort by atom no
            sorted_array = atoms_array[:, atoms_array[1, :].argsort()]
            edge_index = [np.append(sorted_array[1],sorted_array[2]), np.append(sorted_array[2],sorted_array[1])]
            edge_type = np.append(sorted_array[0],sorted_array[0])


            data = Data(x = torch.tensor(sampled_node_symbol_one_hot, dtype=torch.float, **device_kwargs),
                        edge_index=torch.tensor(edge_index, dtype=torch.long, **device_kwargs),
                        edge_type=torch.tensor(edge_type,dtype=torch.float, **device_kwargs),
                        distance = torch.tensor(distance[i], dtype=torch.long, **device_kwargs),
                        node_sequence = torch.tensor(node_sequence[i], dtype=torch.long, **device_kwargs),
                        edge_type_mask = torch.tensor(edge_type_mask[i], dtype=torch.long, **device_kwargs),
                        edge_masks = torch.tensor(edge_mask[i], dtype = torch.long, **device_kwargs),
                        overlapped_edge_features = torch.tensor(overlapped_edge_features[i], dtype = torch.long, **device_kwargs)
                  )
            data_list.append(data)
        return data_list 