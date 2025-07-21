#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
轨迹相似度模型模块
实现了基于GCN、Transformer和注意力机制的轨迹编码器
"""

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
    def __init__(self, activation, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)

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


class SMNEncoder(nn.Module):
    """
    时空多模态编码器
    结合GCN、Transformer和注意力机制
    """
    def __init__(self, hidden_size, dropout=0.1):
        """
        初始化编码器
        参数:
            hidden_dim: 隐藏层维度
            dropout: Dropout比率
        """
        super(SMNEncoder, self).__init__()
        self.hidden_size = hidden_size
        # 初始化各层
        self.mlp_ele = torch.nn.Linear(2, int(hidden_size / 2)).cuda()
        self.mlp_ele_t = torch.nn.Linear(1, hidden_size).cuda()
        self.st = torch.nn.Linear(int(2*hidden_size), hidden_size).cuda()
        
        self.time_embedding = Time2Vec('sin', hidden_size)
        self.gcn_model = GCN_model(int(hidden_size / 2), hidden_size, int(hidden_size / 2)).cuda()
        
        # Transformer编码器
        self.spatial_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(hidden_size, nhead=4, dropout=dropout),
            num_layers=config.num_layers
        ).cuda()
        self.w_omega_s = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.u_omega_s = nn.Parameter(torch.Tensor(hidden_size, 1))
        nn.init.uniform_(self.w_omega_s, -0.1, 0.1)
        nn.init.uniform_(self.u_omega_s, -0.1, 0.1)

        self.temporal_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(hidden_size, nhead=4, dropout=dropout),
            num_layers=args.num_layers
        ).cuda()
        self.w_omega_t = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.u_omega_t = nn.Parameter(torch.Tensor(hidden_size, 1))
        nn.init.uniform_(self.w_omega_t, -0.1, 0.1)
        nn.init.uniform_(self.u_omega_t, -0.1, 0.1)
        
    def forward(self, inputs, network_data):
        """
        前向传播
        参数:
            inputs: 输入数据
            network_data: 路网数据
        返回:
            轨迹编码
        """
        input, input_len = inputs
        
        # GCN处理
        if args.use_GCN:
            logger.debug('使用GCN处理路网数据')
            x = self.mlp_ele(network_data.x)
            graph_node_embeddings = self.gcn_model(x, network_data.edge_index, network_data.edge_weight)
            mlp_input = graph_node_embeddings[input]
        else:
            logger.debug('使用MLP处理输入数据')
            mlp_input = self.nonLeaky(self.mlp_ele(input))
            
        # 创建掩码
        mask = (input[:,:,0] != 0).unsqueeze(-2).cuda()

        # Transformer处理
        output = self.seq_model(mlp_input, mask)
        
        # 时空注意力处理
        traj_output = self.ST_layer(output, mask)
        return traj_output


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

