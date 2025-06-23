import os
import torch
import torch.nn.functional as F
import tqdm
import numpy as np
import logging
import torch.optim as opti
import random
from torch_geometric.data import Data
from pars_args import args
from torch.utils.data import Dataset



class Trainer:
    """
    模型训练类，负责模型的训练、验证和评估
    
    主要功能：
    1. 模型训练和验证
    2. 模型评估和指标计算
    3. 模型保存和加载
    
    属性:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        eva_train: 评估训练数据
        emb_train: 训练数据嵌入
        rec: 召回率记录
        distance: 距离矩阵
        k: 评估指标k值列表
        n_epochs: 训练轮数
        lr: 学习率
        save_epoch_int: 模型保存间隔
        model_folder: 模型保存路径
        device: 训练设备
        model: 模型实例
        is_train: 是否训练模式
        tao: 温度参数
    """
    def __init__(self, model, train_dataloader, val_dataloader, eva_train, embedding_train, recall, distance,
                 lr, n_epochs, device, save_epoch_int, model_folder, is_train=True):
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.eva_train = eva_train
        self.emb_train = embedding_train
        self.rec = recall
        self.distance = distance
        self.k = [1, 5, 10, 20, 50]
        self.n_epochs = n_epochs
        self.lr = lr
        self.save_epoch_int = save_epoch_int
        self.model_folder = model_folder
        self.device = device
        self.model = model
        self.is_train = is_train
        self.tao = 0.05
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        self.opti = opti.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        # self.opti = opti.Adam(
        #     filter(lambda p: p.requires_grad, self.model.parameters()),
        #     lr=lr, weight_decay=0.0005
        # )

        # self.opti = opti.SGD(self.model.parameters(), lr=lr)
        # self.opti = opti.RMSprop(self.model.parameters(), lr=lr, alpha=0.9, weight_decay=0.0005)
        # self.scheduler = opti.lr_scheduler.StepLR(optimizer=self.opti, step_size=5, gamma=0.5)

    def _pass(self, data, train=True):
        """
        单次前向传播
        
        参数:
            data: 输入数据
            train: 是否为训练模式
        返回:
            损失值
        """
        self.opti.zero_grad()

        grid_batch, len_batch = data
        grid_batch = grid_batch.to(self.device)

        output = self.model(grid_batch, len_batch)

        label = torch.arange(0, output.shape[0], device=self.device)
        label = label + 1 - label % 2 * 2
        similarities = F.cosine_similarity(output.unsqueeze(1), output.unsqueeze(0), dim=2)
        similarities = similarities - torch.eye(output.shape[0], device=self.device) * 1e12
        similarities = similarities / self.tao
        loss = F.cross_entropy(similarities, label)
        loss = torch.mean(loss)

        if train:
            loss.backward()
            self.opti.step()
        return loss.item()

    def train_epoch(self):
        """
        训练一个epoch
        
        返回:
            平均损失值
        """
        self.model.train()

        losses = []
        p_bar = tqdm(self.train_loader)
        for data in p_bar:
            loss = self._pass(data)
            losses.append(loss)
            p_bar.set_description('[loss: %f]' % loss)
        # self.scheduler.step()
        return np.array(losses).mean()

    def val_epoch(self):
        """
        验证一个epoch
        
        返回:
            平均损失值
        """
        self.model.eval()

        losses = []
        p_bar = tqdm(self.val_loader)
        for data in p_bar:
            loss = self._pass(data, False)
            losses.append(loss)
            p_bar.set_description('[loss: %f]' % loss)

        return np.array(losses).mean()

    def eva(self):
        """
        模型评估
        
        返回:
            Top-20召回率
        """
        self.model.eval()

        p_bar = tqdm(self.eva_train)
        for batch_id, (grid_batch, len_batch, idx_batch) in enumerate(p_bar):
            grid_batch = grid_batch.to(self.device)
            output = self.model(grid_batch, len_batch)
            self.emb_train[idx_batch, :] = output.detach()
        top_20 = self.test_model(self.emb_train.cpu().numpy(), range(3000, 6000))
        return top_20

    def test_model(self, tra_embeddings, test_range):
        """
        测试模型性能
        
        参数:
            tra_embeddings: 轨迹嵌入
            test_range: 测试范围
        返回:
            Top-10召回率
        """
        top_1_count, top_5_count, top_10_count = 0, 0, 0,
        test_tra_num = 3000
        for i in test_range:
            test_distance = [(j, float(np.sum(np.square(tra_embeddings[i] - e))))
                             for j, e in enumerate(tra_embeddings)]
            true_distance = list(enumerate(self.distance[i][:len(tra_embeddings)]))
            s_test_distance = sorted(test_distance, key=lambda a: a[1])
            s_true_distance = sorted(true_distance, key=lambda a: a[1])

            top_1_recall = [l[0] for l in s_test_distance[:2] if l[0] in [j[0] for j in s_true_distance[:2]]]
            top_5_recall = [l[0] for l in s_test_distance[:6] if l[0] in [j[0] for j in s_true_distance[:6]]]
            top_10_recall = [l[0] for l in s_test_distance[:11] if l[0] in [j[0] for j in s_true_distance[:11]]]

            top_1_count += len(top_1_recall) - 1
            top_5_count += len(top_5_recall) - 1
            top_10_count += len(top_10_recall) - 1

        print('Test on {} trajs'.format(test_tra_num))
        print('Search Top 1 recall {}'.format(float(top_1_count) / (test_tra_num * 1)))
        print('Search Top 5 recall {}'.format(float(top_5_count) / (test_tra_num * 5)))
        print('Search Top 10 recall {}'.format(float(top_10_count) / (test_tra_num * 10)))

        return float(top_10_count) / (test_tra_num * 10)

    def train(self):
        """
        模型训练主循环
        """
        best_acc = 0
        for epoch in range(self.n_epochs):
            if self.is_train:
                train_loss = self.train_epoch()
                val_loss = self.val_epoch()
                logging.info(
                    '[Epoch %d/%d] [training loss: %f] [validation loss: %f]' %
                    (epoch, self.n_epochs, train_loss, val_loss)
                )
                if (epoch + 1) % self.save_epoch_int == 0:
                    save_file = self.model_folder + 'contrastive_epoch_%d.pt' % epoch
                    torch.save(self.model.state_dict(), save_file)
                acc = self.eva()
                if acc > best_acc:
                    best_acc = acc
                    save_file = self.model_folder + 'contrastive_best.pt'
                    torch.save(self.model.state_dict(), save_file)
                    
                    
if __name__ == '__main__':
    print('process ', args.data_set_type)
    data_path = r'processed_' + args.data_set_type + '/'
    model_path = r'saved_models/' + args.data_set_type + '/'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ['CUDA_LAUNCH_BLOCKING'] = args.gpu_id
    np.random.seed(1234)
    torch.manual_seed(1234)
    random.seed(1234)
    setup_logger('train.log')
    if args.data_set_type == 'porto':
        max_grid_size = 39200
        sub_list = [-281, -280, -279, 1, 281, 280, 279, -1]
        node = 11413
    elif args.data_set_type == 'geolife':
        max_grid_size = 1300000
        sub_list = [-1301, -1300, -1299, 1, 1301, 1300, 1299, -1]
        node = 27455
    embedding_size = 128
    grid2id_dict = np.load(data_path + args.data_set_type + '_entities2id.npy', allow_pickle=True).item()
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_set = np.load(data_path + args.data_set_type + '_all_set.npy', allow_pickle=True)[:args.train_num]
    data_set = [x[1] for x in data_set]
    grid_ob = BuildGrid(max_grid_size, embedding_size)
    grid_ob.grid2id_dict = grid2id_dict
    grid_ob.build_data(data_set, data_path)

    entity_embedding = np.load(model_path + 'entity_pair_embedding.npy', allow_pickle=True)[:, args.embedding_dim:]
    entity_embedding = torch.Tensor(entity_embedding).to(d)

    train_loader = contrastive_train_loader(data_path, 128)
    val_loader = contrastive_val_loader(data_path, 128)
    eva_data_train = eva_train_loader(data_path, 128)
    train_distance = np.load(data_path + 'dist_matrix.npy', allow_pickle=True)
    emb_train = torch.zeros((args.train_num+args.val_num, args.tra_dim * 2), device=d, requires_grad=False)
    rec = torch.zeros((args.train_num, 5), device=d, requires_grad=False)
    A_normed = normalize(node).to(d)
    GCN_model = GCN(A_normed, args.embedding_dim, args.embedding_dim)
    GCN_model = GCN_model.to(d)
    entity_embedding = GCN_model(entity_embedding)
    lstm_model = LSTMModel(args.embedding_dim, args.tra_dim, args.num_layers, entity_embedding)
    lstm_model = lstm_model.to(d)
    trainer = Trainer(lstm_model, train_loader, val_loader, eva_data_train, emb_train, rec, train_distance,
                      args.contrastive_lr, args.n_epochs, d, args.save_epoch_int, model_path, True)
    trainer.train()
