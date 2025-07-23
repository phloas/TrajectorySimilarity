import logging
import os

# 模型保存相关配置
saveModel = False

# 模型配置
dim = 128  # 嵌入维度
dropout = 0.1
nheads = 4
num_layers = 1
use_GCN = True  # 是否使用图卷积网络
use_Time_encoder = True  # 是否使用时间编码器

# 三元组损失相关参数
tripleWeight = 0.99  # 三元组损失的权重
triEpoch = -1  # 三元组训练的轮数，-1表示不限制

# 训练参数配置
method_name = "matching"  # 可选方法: "neutraj", "matching", "t2s", "t3s", "srn"
GPU = "0"  # 使用的GPU编号
learning_rate = 0.001  # 学习率
seeds_radio = 0.2  # 训练集比例
epochs = 100  # 训练轮数
batch_size = 20  # 批次大小
sampling_num = 10  # 采样数量

# 数据集配置
data_type = 'beijing'  # 可选数据集: 'porto', 'beijing'
BEIJING_LAT_RANGE = [39.6, 40.7] # 北京纬度范围
BEIJING_LON_RANGE = [115.9, 117.0] # 北京经度范围

PORTO_LON_RANGE = [-9.0, -7.9]
PORTO_LAT_RANGE = [40.7, 41.8]

# 距离度量方法配置
distance_type = 'dtw'  # 可选距离度量: 'hausdorff', 'dtw', 'discret_frechet', 'lcss', 'edr', 'erp'

# 根据不同的距离度量方法设置相应的权重和损失函数
if distance_type == 'lcss':
    disWeight = 0.9  # LCSS距离的权重
    tripleLoss = False  # 是否使用三元组损失
else:
    disWeight = 0.5  # 其他距离度量的权重
    tripleLoss = True  # 使用三元组损失

# 数据文件路径配置
train_set_path = os.path.join('features', data_type, distance_type, 'train_set.npz')  # 训练集距离矩阵
test_set_path = os.path.join('features', data_type, distance_type, 'test_set.npz')  # 测试集距离矩阵
node_path = os.path.join('features', data_type, data_type + '_node.csv')  # 节点数据文件
delta_s = 200  # 空间划分的阈值
edge_path = os.path.join('features', data_type, data_type + '_edge_'+str(delta_s)+'.csv')  # 边数据文件

# 轨迹数据文件路径
train_traj_path = os.path.join('features', data_type, data_type+'_train_traj_list.npz')  # 训练轨迹数据
test_traj_path = os.path.join('features', data_type, data_type+'_test_traj_list.npz')  # 测试轨迹数据

# 根据不同距离度量方法设置预处理程度
if distance_type == 'dtw':
    mail_pre_degree = 1
elif distance_type == 'lcss':
    mail_pre_degree = 1
elif distance_type == 'erp':
    mail_pre_degree = 1
elif distance_type == 'hausdorff':
    if data_type == 'porto':
        mail_pre_degree = 8
    elif data_type == 'beijing':
        mail_pre_degree = 8
else:
    mail_pre_degree = 1

# 数据集大小配置
if data_type == 'porto':
    datalength = 10000  # Porto数据集大小
    em_batch = 1000  # 嵌入批次大小
if data_type == 'beijing':
    datalength = 9000  # Beijing数据集大小
    em_batch = 900  # 嵌入批次大小
test_num = 8000  # 测试集大小

def config_to_str():
    """
    将配置参数转换为字符串形式，用于日志记录
    返回: 包含所有重要配置参数的格式化字符串
    """
    configs = 'learning_rate = {} '.format(learning_rate) + '\n' + \
              'mail_pre_degree = {} '.format(mail_pre_degree) + '\n' + \
              'training_ratio = {} '.format(seeds_radio) + '\n' + \
              'embedding_size = {}'.format(d) + '\n' + \
              'epochs = {} '.format(epochs) + '\n' + \
              'datatype = {} '.format(data_type) + '\n' + \
              'distance_type = {}'.format(distance_type) + '\n' + \
              'batch_size = {} '.format(batch_size) + '\n' + \
              'sampling_num = {} '.format(sampling_num) + '\n' + \
              'tripleLoss = {}'.format(tripleLoss) + '\n' + \
              'tripleWeight = {}'.format(tripleWeight) + '\n' + \
              'GPU = {}'.format(GPU) + '\n' + \
              'use GCN = {}'.format(use_GCN) + '\n' + \
              'use TIME = {}'.format(use_Time_encoder) + '\n' + \
              'delta_s = {}'.format(delta_s) + '\n' + \
              'disWeight = {}'.format(disWeight)
    return configs


def setup_logger(fname=None):
    """
    设置日志记录器
    参数:
        fname: 日志文件名，如果为None则只输出到控制台
    """
    if not logging.root.hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M',
            filename=fname,
            filemode='w'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def zipdir(path, ziph, include_format):
    """
    将指定目录下的文件压缩到zip文件中
    参数:
        path: 要压缩的目录路径
        ziph: zip文件对象
        include_format: 要包含的文件格式列表
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1] in include_format:
                filename = os.path.join(root, file)
                arcname = os.path.relpath(os.path.join(root, file), os.path.join(path, '..'))
                ziph.write(filename, arcname)
