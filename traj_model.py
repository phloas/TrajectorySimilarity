#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
轨迹相似度模型模块
实现了基于GCN、Transformer和注意力机制的轨迹编码器
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from tools import config
from torch_geometric.nn import GCNConv

# 配置日志记录器
logger = logging.getLogger(__name__)

class GCN_model(torch.nn.Module):
    """
    图卷积网络模型
    用于处理路网数据的空间特征
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        初始化GCN模型
        参数:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层特征维度
            out_channels: 输出特征维度
        """
        super(GCN_model, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, normalize=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, normalize=True)
        logger.debug(f'初始化GCN模型: 输入维度={in_channels}, 隐藏维度={hidden_channels}, 输出维度={out_channels}')

    def forward(self, x, edge_index, edge_weight):
        """
        前向传播
        参数:
            x: 节点特征
            edge_index: 边索引
            edge_weight: 边权重
        返回:
            更新后的节点特征
        """
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = torch.nn.Linear(1, 1).cuda()
        self.w = torch.nn.Linear(in_features, out_features-1).cuda()
        self.f = torch.sin

    def forward(self, tau):
        v1 = self.f(self.w(tau))
        v2 = self.w0(tau)
        return torch.cat([v1, v2], dim=-1)
class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = torch.nn.Linear(1, 1).cuda()
        self.w = torch.nn.Linear(in_features, out_features-1).cuda()
        self.f = torch.cos

    def forward(self, tau):
        v1 = self.f(self.w(tau))
        v2 = self.w0(tau)
        return torch.cat([v1, v2], dim=-1)
class Time2Vec(nn.Module):
    def __init__(self, activation, dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, dim)

    def forward(self, x):
        x = self.l1(x)
        return x
    
class ST_layer(nn.Module):
    """
    时空注意力层
    用于捕获轨迹的时空特征
    """
    def __init__(self, embedding_dim):
        """
        初始化时空注意力层
        参数:
            embedding_dim: 嵌入维度
        """
        super(ST_layer, self).__init__()
        self.w_1 = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.w_2 = nn.Parameter(torch.Tensor(embedding_dim, 1))
        
        # 初始化权重
        nn.init.uniform_(self.w_1, -0.1, 0.1)
        nn.init.uniform_(self.w_2, -0.1, 0.1)
        logger.debug(f'初始化时空注意力层: 嵌入维度={embedding_dim}')
        
    def getMask(self, seq_lengths):
        """
        创建序列掩码
        参数:
            seq_lengths: 序列长度
        返回:
            掩码张量
        """
        max_len = int(seq_lengths.max())
        mask = torch.ones((seq_lengths.size()[0], max_len)).to(self.device)

        for i, l in enumerate(seq_lengths):
            if l < max_len:
                mask[i, l:] = 0

        return mask
    
    def forward(self, input, mask):
        """
        前向传播
        参数:
            input: 输入特征
            mask: 掩码
        返回:
            加权后的特征表示
        """
        # 计算注意力分数
        v = torch.tanh(torch.matmul(input, self.w_1))
        mask = mask.squeeze(1).squeeze(1)
        att = torch.matmul(v, self.w_2).squeeze()

        # 应用掩码
        if mask is not None:
            att = att.masked_fill(mask == 0, -1e10)

        # 计算注意力权重
        att_score = F.softmax(att, dim=1).unsqueeze(2)
        scored_outputs = input * att_score

        # 加权求和
        out = torch.sum(scored_outputs, dim=1)
        return out

# class SMNTrainEncoder()

# class SMNTestEncoder()
class CrossEncoder(nn.Module):
    def __init__(self, d_model, num_layers, nheads, dropout=config.dropout):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nheads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  # [CLS] token

    def forward(self, emb1, emb2):
        # emb1, emb2: [B, D]
        B, D = emb1.shape
        # 拼接 + [CLS]
        pair = torch.cat([emb1.unsqueeze(1), emb2.unsqueeze(1)], dim=1)  # [B, 2, D]
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        seq = torch.cat([pair, cls_token], dim=1)  # [B, 3, D]6
        out = self.encoder(seq)  # [B, 3, D]
        return out[:, -1, :]  # 取 [CLS] token 的输出

class TrajectoryEncoder(nn.Module):
    def __init__(self, dim, num_layers, nheads, len_threshold=30):
        super().__init__()
        
        # 初始化各层
        self.mlp_ele = torch.nn.Linear(2, int(dim / 2)).cuda()
        self.mlp_ele_t = torch.nn.Linear(1, dim).cuda()
        self.st = torch.nn.Linear(int(2 * dim), dim).cuda()
        self.time_embedding = Time2Vec('sin', dim)
        self.gcn = GCN_model(int(dim / 2), dim, int(dim / 2)).cuda()

        self.len_threshold = len_threshold
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=nheads, batch_first=True),
            num_layers=num_layers
        )
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.fc = nn.Linear(dim, dim)  # GRU输出到traj_emb_dim
        self.traj_emb_dim = dim

    def forward(self, traj_node_ids, network_data):
        # traj_node_ids: [B, L]  (node id序列)
        # network_data.x: [N, node_emb_dim]
        node_embs = network_data.x[traj_node_ids]  # [B, L, node_emb_dim]
        lens = (traj_node_ids != 0).sum(dim=1)  # 假设0为padding
        traj_embs = []
        for i in range(node_embs.size(0)):
            seq = node_embs[i, :lens[i]]
            if lens[i] > self.len_threshold:
                # Transformer
                seq = seq.unsqueeze(0)  # [1, L, D]
                out = self.transformer(seq)  # [1, L, D]
                emb = out[:, 0, :]  # 取第一个token或池化
            else:
                # GRU
                seq = seq.unsqueeze(0)
                _, h = self.gru(seq)  # h: [1, 1, H]
                emb = self.fc(h[-1])  # [1, traj_emb_dim]
            traj_embs.append(emb)
        traj_embs = torch.cat(traj_embs, dim=0)  # [B, traj_emb_dim]
        return traj_embs

class TrajSimilarityTrainModel(nn.Module):
    def __init__(self, dim, num_layers, nheads, len_threshold=30):
        super().__init__()
        self.encoder = TrajectoryEncoder(dim, num_layers, nheads, len_threshold=len_threshold)
        self.cross_encoder = CrossEncoder(dim)
        self.sim_head = nn.Linear(dim, 1)

    def forward(self, traj1_ids, traj2_ids, network_data):
        # traj1_ids, traj2_ids: [B, L]
        emb1 = self.encoder(traj1_ids, network_data)  # [B, D]
        emb2 = self.encoder(traj2_ids, network_data)  # [B, D]
        cross_emb = self.cross_encoder(emb1, emb2)   # [B, D]
        sim_score = self.sim_head(cross_emb).squeeze(-1)  # [B]
        return sim_score, cross_emb, emb1, emb2

# class SMNEncoder(nn.Module):
#     """
#     时空多模态编码器
#     结合GCN、Transformer和注意力机制
#     1. GCN
#     2. Transformer
#     3. 训练用 + Cross
#     """
#     def __init__(self, hidden_size, dropout=0.1):
#         """
#         初始化编码器
#         参数:
#             hidden_dim: 隐藏层维度
#             dropout: Dropout比率
#         """
#         super(SMNEncoder, self).__init__()
#         self.hidden_size = hidden_size
#         # 初始化各层
#         self.mlp_ele = torch.nn.Linear(2, int(hidden_size / 2)).cuda()
#         self.mlp_ele_t = torch.nn.Linear(1, hidden_size).cuda()
#         self.st = torch.nn.Linear(int(2 * hidden_size), hidden_size).cuda()
        
#         self.time_embedding = Time2Vec('sin', hidden_size)
#         self.gcn_model = GCN_model(int(hidden_size / 2), hidden_size, int(hidden_size / 2)).cuda()
        
#         # Transformer spatial编码器
#         self.spatial_encoder = torch.nn.TransformerEncoder(
#             torch.nn.TransformerEncoderLayer(hidden_size, nhead=config.nheads, dropout=dropout),
#             num_layers=config.num_layers
#         ).cuda()
#         self.w_omega_s = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
#         self.u_omega_s = nn.Parameter(torch.Tensor(hidden_size, 1))
#         nn.init.uniform_(self.w_omega_s, -0.1, 0.1)
#         nn.init.uniform_(self.u_omega_s, -0.1, 0.1)

#         # Transformer temporal编码器
#         self.temporal_encoder = torch.nn.TransformerEncoder(
#             torch.nn.TransformerEncoderLayer(hidden_size, nhead=config.nheads, dropout=dropout),
#             num_layers=config.num_layers
#         ).cuda()
#         self.w_omega_t = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
#         self.u_omega_t = nn.Parameter(torch.Tensor(hidden_size, 1))
#         nn.init.uniform_(self.w_omega_t, -0.1, 0.1)
#         nn.init.uniform_(self.u_omega_t, -0.1, 0.1)
        
#     def forward(self, inputs, network_data):
#         """
#         前向传播
#         参数:
#             inputs: 输入数据
#             network_data: 路网数据
#         返回:
#             轨迹编码
#         """
#         input, input_len = inputs
        
#         # GCN处理
#         if config.use_GCN:
#             logger.debug('使用GCN处理路网数据')
#             x = self.mlp_ele(network_data.x)
#             graph_node_embeddings = self.gcn_model(x, network_data.edge_index, network_data.edge_weight)
#             mlp_input = graph_node_embeddings[input]
#         else:
#             logger.debug('使用MLP处理输入数据')
#             mlp_input = self.nonLeaky(self.mlp_ele(input))

#         # 使用GRU或者Transformer进行处理
            
#         # 创建掩码
#         mask = (input[:,:,0] != 0).unsqueeze(-2).cuda()

#         # Transformer处理
#         output = self.seq_model(mlp_input, mask)
        
#         # 时空注意力处理
#         traj_output = self.ST_layer(output, mask)

#         # 使用Cross Encoder进行拼接
        
#         # 返回
#         return traj_output

class Traj_Network(nn.Module):
    """
    轨迹网络模型
    用于轨迹相似度计算
    """
    def __init__(self, dim, num_layers, nheads, batch_size, sampling_num):
        """
        初始化轨迹网络
        参数:
            target_size: 目标维度
            batch_size: 批次大小
            sampling_num: 采样数量
        """
        super(Traj_Network, self).__init__()
        self.dim = dim
        self.batch_size = batch_size
        self.sampling_num = sampling_num
        self.num_layers = num_layers
        self.nheads = nheads
        self.model = TrajSimilarityTrainModel(dim, num_layers, nheads)
        logger.info(f'初始化轨迹网络: 目标维度={dim}, 批次大小={batch_size}, 采样数量={sampling_num}')

    def forward(self, inputs_arrays, time_arrays, inputs_len_arrays, network_data):
        """
        前向传播
        参数:
            inputs_arrays: 输入数组
            time_arrays: 时间数组
            inputs_len_arrays: 输入长度数组
            network_data: 路网数据
        返回:
            轨迹嵌入
        """
        anchor_input = torch.Tensor(inputs_arrays).cuda()
        anchor_input_t = torch.Tensor(time_arrays).cuda()
        anchor_input_len = inputs_len_arrays

        anchor_embedding = self.smn(
            [anchor_input, anchor_input_t, anchor_input_len],
            network_data)

        return anchor_embedding

class Traj_Network(nn.Module):
    """
    轨迹网络模型
    用于轨迹相似度计算
    """
    def __init__(self, target_size, batch_size, sampling_num):
        """
        初始化轨迹网络
        参数:
            target_size: 目标维度
            batch_size: 批次大小
            sampling_num: 采样数量
        """
        super(Traj_Network, self).__init__()
        self.target_size = target_size
        self.batch_size = batch_size
        self.sampling_num = sampling_num
        self.smn = SMNEncoder(self.target_size)
        logger.info(f'初始化轨迹网络: 目标维度={target_size}, 批次大小={batch_size}, 采样数量={sampling_num}')

    def forward(self, inputs_arrays, time_arrays, inputs_len_arrays, network_data):
        """
        前向传播
        参数:
            inputs_arrays: 输入数组
            time_arrays: 时间数组
            inputs_len_arrays: 输入长度数组
            network_data: 路网数据
        返回:
            轨迹嵌入
        """
        anchor_input = torch.Tensor(inputs_arrays).cuda()
        anchor_input_t = torch.Tensor(time_arrays).cuda()
        anchor_input_len = inputs_len_arrays

        anchor_embedding = self.smn(
            [anchor_input, anchor_input_t, anchor_input_len],
            network_data)

        return anchor_embedding

    def matching_forward(self, anchor_input, anchor_time, anchor_input_len, network_data):
        """
        匹配前向传播
        参数:
            anchor_input: 锚点输入
            anchor_time: 锚点时间
            anchor_input_len: 锚点输入长度
            network_data: 路网数据
        返回:
            锚点嵌入、轨迹嵌入和匹配输出
        """
        anchor_embedding, trajs_embedding, outputs_ap, outputs_p = self.smn(
            [autograd.Variable(anchor_input, requires_grad=False).cuda(), anchor_time, anchor_input_len],
            [autograd.Variable(anchor_input, requires_grad=False).cuda(), anchor_time, anchor_input_len],
            network_data)

        return anchor_embedding, trajs_embedding, outputs_ap, outputs_p

