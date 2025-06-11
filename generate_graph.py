import pandas as pd
import pickle
import numpy as np
import os.path as osp
import os # 导入os模块，用于路径操作和目录创建
from tqdm import tqdm
from tools import config  # 假设 config.py 存在并定义了 data_type 和 datalength
import csv
import math
import argparse # 导入argparse模块，用于处理命令行参数


# --- 全局常量和配置 ---
# Geolife数据集的经纬度范围
beijing_lat_range = [39.6, 40.7]
beijing_lon_range = [115.9, 117.1]

# Porto数据集的经纬度范围
porto_lat_range = [40.7, 41.8]
porto_lon_range = [-9.0, -7.9]

# 距离阈值列表，用于生成不同连接密度的图。单位为米。
# 150米阈值对应的是主 edge.csv 文件。
DISTANCE_THRESHOLDS = [100.0, 125.0, 150.0, 175.0, 200.0]

# --- 辅助函数 ---

def haversine(lat1, lng1, lat2, lng2):
    """
    使用Haversine公式计算地球上两点之间的距离。
    
    参数:
    lat1 (float): 第一个点的纬度 (度)
    lng1 (float): 第一个点的经度 (度)
    lat2 (float): 第二个点的纬度 (度)
    lng2 (float): 第二个点的经度 (度)
    
    返回:
    float: 两点之间的距离，单位为米
    """
    R = 6371 * 1000  # 地球半径，单位为米
    # 将经纬度从度转换为弧度
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])

    dlat = lat2 - lat1
    dlng = lng2 - lng1
    
    # Haversine公式计算球面距离
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def write_edges_to_csv(edges, threshold, data_type):
    """
    将生成的边数据写入CSV文件。
    
    参数:
    edges (list): 包含边的列表，每条边为 [起始节点ID, 结束节点ID, 长度]
    threshold (float): 用于命名文件的距离阈值。
                       如果为150.0，文件名为 '_edge.csv'；否则为 '_edge_X.csv'。
    data_type (str): 数据集类型 (例如 'beijing', 'porto')
    """
    # 根据阈值构建文件名
    if threshold == 150.0:
        file_name = osp.join('features', data_type, f'{data_type}_edge.csv')
    else:
        file_name = osp.join('features', data_type, f'{data_type}_edge_{int(threshold)}.csv')

    # 确保目标目录存在
    output_dir = osp.join('features', data_type)
    os.makedirs(output_dir, exist_ok=True) # 使用 os.makedirs 确保目录存在

    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['s_node', 'e_node', 'length'])  # 写入CSV文件头
        writer.writerows(edges) # 写入所有边数据
    print(f"边数据 (阈值: {threshold}m) 已写入: {file_name}，共 {len(edges)} 条边。")

# --- 主要功能函数 ---

def extract_nodes_edges(traj_grid, traj_coor, data_type):
    """
    从轨迹数据中提取唯一的网格节点和真实坐标节点，并生成基于距离的边。
    这些边代表了在指定距离阈值内的所有唯一坐标点对。
    
    参数:
    traj_grid (list): 包含每个轨迹的网格点列表。格式如 [[x1, y1], [x2, y2], ...]
    traj_coor (list): 包含每个轨迹的真实坐标点列表 (经度, 纬度)。格式如 [[lng1, lat1], [lng2, lat2], ...]
    data_type (str): 数据集类型 (例如 'beijing', 'porto')
    """
    print("步骤1/3: 开始提取唯一节点并映射ID...")
    traj_id = []      # 存储每个轨迹的节点ID序列
    grid2id = {}      # 网格坐标 (tuple) 到节点ID的映射
    coor2id = {}      # 真实坐标 (tuple(lng, lat)) 到节点ID的映射 (用于边的生成)
    node_id = 0       # 节点ID计数器，从0开始递增

    # 遍历所有轨迹，为每个唯一的网格点和坐标点分配一个唯一的节点ID
    for grids, coors in tqdm(zip(traj_grid, traj_coor), desc="生成节点ID"):
        tmp_traj = [] # 用于存储当前轨迹中点的节点ID序列
        for grid, coor in zip(grids, coors):
            # 如果当前网格点尚未被记录，则分配新的节点ID
            if tuple(grid) not in grid2id:
                grid2id[tuple(grid)] = node_id
                coor2id[tuple([coor[0], coor[1]])] = node_id # 记录真实坐标与节点ID的映射
                node_id += 1 # 节点ID递增
            tmp_traj.append(grid2id[tuple(grid)]) # 将当前点的节点ID添加到临时轨迹列表
        traj_id.append(tmp_traj) # 将处理后的轨迹节点ID序列添加到总列表
    print(f"步骤1/3: 节点ID映射完成。共识别出 {node_id} 个唯一节点。")

    # 可选：保存轨迹的节点ID序列到pickle文件
    # with open(osp.join('./features', data_type, f'{data_type}_traj_id'), 'wb') as f:
    #     pickle.dump(traj_id, f)
    # print(f"轨迹ID序列已保存到: ./features/{data_type}/{data_type}_traj_id")

    # 步骤2: 计算所有唯一坐标点之间的Haversine距离，并收集所有潜在的边
    print("步骤2/3: 开始计算节点间的Haversine距离并收集潜在边...")
    all_potential_edges = [] # 存储所有计算出的边 (s_node_id, e_node_id, distance)
    unique_points = list(coor2id.keys()) # 获取所有唯一的真实坐标点
    unique_point_ids = list(coor2id.values()) # 获取对应的节点ID

    # 遍历所有唯一的坐标点对，计算它们之间的距离
    # 使用双层循环，只计算一次 (i, j) 对，并添加双向边
    for i in tqdm(range(len(unique_points)), desc="计算点对距离"):
        point1 = unique_points[i]
        point1_id = unique_point_ids[i]
        for j in range(i + 1, len(unique_points)): # 避免重复计算和自身到自身的距离
            point2 = unique_points[j]
            point2_id = unique_point_ids[j]
            
            distance = haversine(point1[0], point1[1], point2[0], point2[1])
            
            # 如果距离不为0，则添加这条潜在的边及其反向边，表示无向图
            if distance > 0.0:
                all_potential_edges.append([point1_id, point2_id, distance])
                all_potential_edges.append([point2_id, point1_id, distance]) # 添加反向边

    print(f"步骤2/3: 已收集 {len(all_potential_edges)} 条潜在边 (包含双向边)。")

    # 步骤3: 根据不同的距离阈值筛选并写入边文件
    print("步骤3/3: 根据距离阈值筛选边并写入CSV文件...")
    for threshold in DISTANCE_THRESHOLDS:
        edges_for_threshold = []
        for s_node, e_node, dist in all_potential_edges:
            if dist <= threshold:
                edges_for_threshold.append([s_node, e_node, dist])
        write_edges_to_csv(edges_for_threshold, threshold, data_type)

    print("节点和边提取完成。")

def extract_edges_from_trajectory(trajectory, nodes_dict):
    """
    此函数用于从单个轨迹中提取连续的边，并统计其出现次数。
    与 `extract_nodes_edges` 不同，此函数关注的是轨迹中连续点之间的连接，
    而非所有空间上接近的点之间的连接。
    
    参数:
    trajectory (list): 包含轨迹点坐标的列表 (经度, 纬度)。
                       例如：[(lng1, lat1), (lng2, lat2), ...]
    nodes_dict (dict): 真实坐标 (tuple) 到节点ID的映射，用于将坐标转换为ID。
    
    返回:
    list: 包含唯一边的列表，每条边为 [起始节点ID, 结束节点ID, 长度, 计数]。
          边的方向被规范化 (s_node <= e_node) 以便去重和计数。
    """
    edges_in_trajectory = set()  # 使用集合来去重 (s_node_id, e_node_id, length)
    print("开始从单个轨迹中提取连续边...")
    # 遍历轨迹中的相邻点对
    for i in range(1, len(trajectory)):
        lat1, lng1 = trajectory[i - 1]
        lat2, lng2 = trajectory[i]

        # 使用节点字典将经纬度转换为节点 ID
        # 注意：这里使用 get() 来处理可能不存在的键，并返回 None
        s_node = nodes_dict.get(tuple([lng1, lat1]))
        e_node = nodes_dict.get(tuple([lng2, lat2]))

        if s_node is not None and e_node is not None:
            length = haversine(lat1, lng1, lat2, lng2)  # 计算两点之间的距离
            # 规范化边的方向 (小ID在前)，以便在集合中正确去重无向边
            # 同时保留长度作为边的一部分特征
            normalized_edge = tuple(sorted([s_node, e_node])) + (length,)
            edges_in_trajectory.add(normalized_edge)

    # 统计每条唯一边（包括方向和长度）的出现次数
    unique_edges = []
    edge_count = {}
    for s_node, e_node, length in edges_in_trajectory:
        # key 包含起点、终点和长度，确保统计的是完全相同的边（即使方向不同，如果已规范化则视为同条）
        edge_key = (s_node, e_node, length) 
        edge_count[edge_key] = edge_count.get(edge_key, 0) + 1
    
    # 将统计结果转换为列表形式
    for (s_node, e_node, length), count in edge_count.items():
        unique_edges.append([s_node, e_node, length, count])

    print(f"单个轨迹边提取完成。共识别出 {len(unique_edges)} 条唯一连续边。")
    return unique_edges

def main():
    parser = argparse.ArgumentParser(description='Generate trajectory graphs based on distance thresholds.')

    # 添加命令行参数
    parser.add_argument('--dataset', type=str, required=True, choices=['geolife', 'porto'],
                        help='Dataset to process (e.g., geolife or porto)')
    parser.add_argument('--num_traj', type=int, default=None,
                        help='Number of trajectories to process. If not specified, all available trajectories will be used.')

    # 解析命令行参数
    args = parser.parse_args()

    print(f"脚本开始执行：生成 {args.dataset} 轨迹图。")

    if args.dataset == 'geolife':
        data_type_name = 'geolife'
        default_num_traj = 9000 # Geolife的默认轨迹数量
    elif args.dataset == 'porto':
        data_type_name = 'porto'
        default_num_traj = 10000 # Porto的默认轨迹数量
    else:
        raise ValueError("Invalid dataset specified. Choose 'geolife' or 'porto'.")

    num_traj_to_process = args.num_traj if args.num_traj is not None else default_num_traj

    # 从pickle文件加载预处理后的轨迹坐标和网格数据
    try:
        coor_file_path = osp.join('features', data_type_name, f'{data_type_name}_traj_coord')
        grid_file_path = osp.join('features', data_type_name, f'{data_type_name}_traj_grid')
        
        # 加载数据并根据 num_traj_to_process 进行切片
        all_coor_list_int = np.array(pickle.load(open(coor_file_path, 'rb'))[0], dtype=object)[:num_traj_to_process]
        all_grid_list_int = np.array(pickle.load(open(grid_file_path, 'rb'))[0], dtype=object)[:num_traj_to_process]
        
        print(f"成功加载 {len(all_coor_list_int)} 条 {data_type_name} 轨迹的坐标和网格数据。")
    except FileNotFoundError:
        print(f"错误：未找到特征文件。请确保以下路径存在：")
        print(f"  - {coor_file_path}")
        print(f"  - {grid_file_path}")
        print(f"请先运行 preprocess.py 为数据集 '{data_type_name}' 生成这些特征文件。")
        exit(1) # 脚本异常退出
    except Exception as e:
        print(f"加载轨迹特征文件时发生意外错误: {e}")
        exit(1)

    # 调用核心函数提取节点和生成边
    extract_nodes_edges(all_grid_list_int, all_coor_list_int, data_type_name)

    # 原始代码中对一个可能存在的 'config.edge_path' 文件进行的额外过滤。
    # 如果 'extract_nodes_edges' 已经生成了所有需要的边文件，
    # 并且其中包含了一个名为 '_edge.csv' (对应150m阈值) 的文件，
    # 那么这部分代码可能是多余的，或者其作用是处理一个预先存在的、未按阈值分类的全局边文件。
    # 为了避免重复写入，且考虑到 extract_nodes_edges 已经生成了带有150m阈值的 _edge.csv，
    # 暂时将此段代码注释掉。如果需要额外的过滤逻辑，应明确其作用和输入。
    # print("开始对原始边文件进行额外过滤 (长度 <= 150m)，如果存在的话...")
    # try:
    #     df_edge = pd.read_csv(config.edge_path, sep=',')
    #     df_edge_filtered = df_edge[df_edge['length'] <= 150]
    #     output_path_150_recheck = osp.join('./features', data_type_name, f'{data_type_name}_edge.csv')
    #     df_edge_filtered.to_csv(output_path_150_recheck, index=False)
    #     print(f"已过滤并保存长度 <= 150m 的边到: {output_path_150_recheck}")
    # except FileNotFoundError:
    #     print(f"警告：未找到 config.edge_path 指定的原始边文件 '{config.edge_path}'。跳过额外过滤。")
    # except Exception as e:
    #     print(f"过滤边文件时发生错误: {e}")

    print("脚本执行结束。")


if __name__ == '__main__':
    main()