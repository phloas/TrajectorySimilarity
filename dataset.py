"""
dataset.py

该模块负责加载预处理后的轨迹数据，并根据空间距离和时间距离（如果启用）生成用于训练和测试的相似性标签。
它还会将轨迹数据和生成的标签保存为NumPy的.npz文件，供后续模型训练使用。

主要功能：
-   从特征文件中加载轨迹坐标和时间信息。
-   加载预先计算好的空间距离矩阵和时间距离矩阵。
-   根据距离矩阵计算轨迹相似性标签（近邻轨迹）。
-   将数据集划分为训练集和测试集。
-   保存处理后的轨迹数据和标签。
"""

import pickle
import numpy as np
import os.path as osp
from tqdm import tqdm
from tools import config
from triplet_sampling import get_train_triplet_label


def get_label(input_dis_matrix, input_time_matrix, count):
    """
    根据综合距离矩阵为测试集轨迹生成相似性标签。
    对于每条轨迹，找到与其距离最近的 `count` 条轨迹的索引。

    参数:
    input_dis_matrix (np.ndarray): 空间距离矩阵。
    input_time_matrix (np.ndarray): 时间距离矩阵。
    count (int): 查找的最近邻轨迹数量。

    返回值:
    np.ndarray: 包含每条轨迹最近邻索引的数组。
    """
    print("正在为测试集生成相似性标签...")
    label = []
    for i in tqdm(range(len(input_dis_matrix)), desc="生成标签"):
        input_r = np.array(input_dis_matrix[i])
        input_t = np.array(input_time_matrix[i])
        out = config.disWeight*input_r+(1-config.disWeight)*input_t
        idx = np.argsort(out)
        label.append(idx[1:count+1])
    print("测试集相似性标签生成完成。")
    return np.array(label, dtype=object)


train_size = int(config.datalength*config.seeds_radio)
test_size = int(config.datalength)

print(f"配置数据长度: {config.datalength}, 训练集比例: {config.seeds_radio}")
print(f"计算得到的训练集大小: {train_size}, 测试集大小: {test_size}")

all_list_int = np.array(pickle.load(
    open(osp.join('features', config.data_type+'_traj_coord'), 'rb'))[0], dtype=object)[:config.datalength]

all_id_list_int = np.array(pickle.load(
    open(osp.join('features', config.data_type, config.data_type+'_traj_id'), 'rb')), dtype=object)[:config.datalength]

print(f"当前数据集类型: {config.data_type}")
print(f"当前距离类型: {config.distance_type}")

beijing_lat_range = [39.6, 40.7]
beijing_lon_range = [115.9, 117.1]

porto_lat_range = [40.7, 41.8]
porto_lon_range = [-9.0, -7.9]

all_coor_list_int = []
all_time_list_int = []
print("正在分离轨迹的坐标和时间信息...")
for trajs in tqdm(all_list_int, desc="分离坐标与时间"):
    tmp_coor = []
    tmp_time = []
    for lat, lng, timeslot, times in trajs:
        tmp_coor.append([lat, lng])
        tmp_time.append([timeslot, times])
    all_coor_list_int.append(tmp_coor)
    all_time_list_int.append(tmp_time)
all_coor_list_int = np.array(all_coor_list_int, dtype=object)
all_time_list_int = np.array(all_time_list_int, dtype=object)
print(f"成功分离 {len(all_coor_list_int)} 条轨迹的坐标和时间信息。")

train_coor_list = all_coor_list_int[0:train_size]
test_coor_list = all_coor_list_int[train_size:test_size]

train_time_list = all_time_list_int[0:train_size]
test_time_list = all_time_list_int[train_size:test_size]

train_id_list = all_id_list_int[0:train_size]
test_id_list = all_id_list_int[train_size:test_size]
print(f"训练集轨迹数量: {len(train_coor_list)}, 测试集轨迹数量: {len(test_coor_list)}。")

dis_matrix = np.load(osp.join('ground_truth', config.data_type,
                              config.distance_type, config.distance_type + '_spatial_distance.npy'))
time_matrix = np.load(osp.join('ground_truth', config.data_type,
                               config.distance_type, config.distance_type + '_temporal_distance.npy'))

print("正在对距离矩阵进行后处理 (对角线归零并归一化)...")
np.fill_diagonal(dis_matrix, 0)
np.fill_diagonal(time_matrix, 0)

if np.max(dis_matrix) != 0:
    dis_matrix = dis_matrix/np.max(dis_matrix)
else:
    print("空间距离矩阵最大值为0，跳过归一化。")

if np.max(time_matrix) != 0:
    time_matrix = time_matrix/np.max(time_matrix)
else:
    print("时间距离矩阵最大值为0，跳过归一化。")

train_dis_matrix = dis_matrix[0:train_size, 0:train_size]
test_dis_matrix = dis_matrix[train_size:test_size, train_size:test_size]

train_time_matrix = time_matrix[0:train_size, 0:train_size]
test_time_matrix = time_matrix[train_size:test_size, train_size:test_size]

print("正在为训练集生成正负样本标签 (10个正样本)...")
train_y, train_neg_y, train_dis, train_neg_dis = get_train_triplet_label(train_dis_matrix, train_time_matrix, 10)
print("正在为测试集生成标签 (50个最近邻)...")
test_y = get_label(test_dis_matrix, test_time_matrix, 50)
print("标签生成完成。")

print("正在保存训练集轨迹数据到: " + config.train_traj_path)
try:
    np.savez(config.train_traj_path,
             coor=train_coor_list,
             id=train_id_list,
             time=train_time_list)
    print("训练集轨迹数据保存成功。")
except Exception as e:
    print(f"保存训练集轨迹数据时发生错误: {e}")

print("正在保存训练集标签和距离矩阵数据到: " + config.train_set_path)
try:
    np.savez(config.train_set_path,
             train_y=train_y,
             train_neg_y=train_neg_y,
             train_dis=train_dis,
             train_neg_dis=train_neg_dis,
             train_dis_matrix=train_dis_matrix,
             train_time_matrix=train_time_matrix)
    print("训练集标签和距离矩阵数据保存成功。")
except Exception as e:
    print(f"保存训练集标签和距离矩阵数据时发生错误: {e}")

print("正在保存测试集轨迹数据到: " + config.test_traj_path)
try:
    np.savez(config.test_traj_path,
             coor=test_coor_list,
             id=test_id_list,
             time=test_time_list)
    print("测试集轨迹数据保存成功。")
except Exception as e:
    print(f"保存测试集轨迹数据时发生错误: {e}")

print("正在保存测试集标签数据到: " + config.test_set_path)
try:
    np.savez(config.test_set_path,
             label_idx=test_y)
    print("测试集标签数据保存成功。")
except Exception as e:
    print(f"保存测试集标签数据时发生错误: {e}")

print("dataset.py 脚本执行完毕。")
