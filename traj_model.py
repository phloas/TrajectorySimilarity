import torch
import torch.nn as nn
import torch.nn.functional as F
from pars_args import args
from torch_geometric.nn import GCNConv
"""
SMN Encoder
GCN + Transformer + AttentionPooling
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
    
class ST_layer(nn.Module):
    def __init__(self, embedding_dim):
        super(ST_layer, self).__init__()
        self.w_1 = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.w_2 = nn.Parameter(torch.Tensor(embedding_dim, 1))
        
        nn.init.uniform_(self.w_1, -0.1, 0.1)
        nn.init.uniform_(self.w_2, -0.1, 0.1)
        
    def getMask(self, seq_lengths):
        """
        create mask based on the sentence lengths
        :param seq_lengths: sequence length after `pad_packed_sequence`
        :return: mask (batch_size, max_seq_len)
        """
        max_len = int(seq_lengths.max())

        # (batch_size, max_seq_len)
        mask = torch.ones((seq_lengths.size()[0], max_len)).to(self.device)

        for i, l in enumerate(seq_lengths):
            if l < max_len:
                mask[i, l:] = 0

        return mask
    
    def forward(self, input, mask):
        # [batch_size, seq_len, d_model]
        # [batch_size, 1, 1, seq_len]
        # Attention...
        # (batch_size, seq_len, 2 * num_hiddens)
        v = torch.tanh(torch.matmul(input, self.w_1))
        mask = mask.squeeze(1).squeeze(1)
        # (batch_size, seq_len)
        att = torch.matmul(v, self.w_2).squeeze()

        # add mask
        if mask is not None:
            att = att.masked_fill(mask == 0, -1e10)

        # (batch_size, seq_len,1)
        att_score = F.softmax(att, dim=1).unsqueeze(2)
        # normalization attention weight
        # (batch_size, seq_len, 2 * num_hiddens)
        scored_outputs = input * att_score

        # weighted sum as output
        # (batch_size, 2 * num_hiddens)
        out = torch.sum(scored_outputs, dim=1)
        # print(out.size()) 
        return out


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
        
        self.ST_layer = ST_layer(hidden_dim)
        
    def forward(self, inputs, network_data):
        input, input_len = inputs # [batch_size, traj_len, 2]
        
        # GCN
        if args.use_GCN:
            x = self.mlp_ele(network_data.x) #[node_num, 2]
            graph_node_embeddings = self.gcn_model(x, network_data.edge_index, network_data.edge_weight) # [node_num, 64]
            mlp_input = graph_node_embeddings[input]
        else:
            mlp_input = self.nonLeaky(self.mlp_ele(input))
            
        # mask
        mask = (input[:,:,0] != 0).unsqueeze(-2).cuda() #[batch_size, 1, traj_len] 

        # Transformer
        output = self.seq_model(mlp_input, mask)
        
        traj_output = self.ST_layer(output, mask)
        return traj_output

# class Traj_Network(nn.Module):
