import torch
import torch.nn as nn
import torch.nn.functional as F
from pars_args import args
from torch_geometric.nn import GCNConv
"""
SMN Encoder
GCN + Transformer
"""
class GCN_model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN_model, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, normalize=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, normalize=True)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x



class SMNEncoder(nn.Module):
    def __init__(self, hidden_dim, dropout = 0.1):
        super(SMNEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.mlp_ele = torch.nn.Linear(2, int(hidden_dim / 2)).cuda()
        
        self.nonLeaky = F.tanh
        self.gcn_model = GCN_model(int(hidden_dim / 2), hidden_dim, int(hidden_dim / 2)).cuda()
        
        self.seq_model = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(hidden_dim, nhead=4, dropout=dropout),
            num_layers=args.num_layers
        ).cuda()
        
    def forward(self, inputs_a, inputs_b, network_data):
        input_a, input_len_a = inputs_a
        input_b, input_len_b = inputs_b
        
        # GCN
        if args.use_GCN:
            x = self.mlp_ele(network_data.x) #[node_num, 2]
            graph_node_embeddings = self.gcn_model(x, network_data.edge_index, network_data.edge_weight) # [node_num, 64]
            mlp_input_a = graph_node_embeddings[input_a]
            mlp_input_b = graph_node_embeddings[input_b] # [batch_size, traj_len, 64]
        else:
            mlp_input_a = self.nonLeaky(self.mlp_ele(input_a))
            mlp_input_b = self.nonLeaky(self.mlp_ele(input_b))
            
        # mask
        mask_a = (input_a[:,:,0] != 0).unsqueeze(-2).cuda() #[batch_size, 1, traj_len] 
        mask_b = (input_b[:,:,0] != 0).unsqueeze(-2).cuda()

        # Transformer
        output_a = self.seq_model(mlp_input_a, mask_a)
        output_b = self.seq_model(mlp_input_b, mask_b)
