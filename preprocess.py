"""
preprocess.py

该脚本用于对轨迹数据进行预处理，并计算轨迹之间的距离。它支持通过命令行参数指定
数据集类型（Geolife或Porto）和距离计算方法。脚本主要包括以下功能：
1.  加载预处理后的轨迹坐标数据。
2.  将轨迹数据转换为NumPy数组格式，并分离坐标和时间信息。
3.  根据用户选择的距离类型（如Hausdorff, DTW等）和是否考虑时间信息来计算轨迹距离。
4.  通过命令行参数灵活配置处理过程。
"""

import pickle 
import numpy as np
import time
import argparse 
from tools.config import batch_size
# 导入自定义工具模块
from generate_feature import trajectory_feature_generation # 导入用于轨迹特征生成的preprocess模块
from tools.distance_computation import trajecotry_distance_list_time, trajecotry_distance_list # 导入距离计算函数


def distance_comp(coor_path, num_traj, distance_type, data_type, use_time_in_distance=False):
    traj_coord = pickle.load(open(coor_path, 'rb'))[0][:num_traj]
    np_traj_coord = []
    np_traj_time = []
    for t in traj_coord:
        temp_coord = []
        temp_time = []
        for item in t:
            temp_coord.append([item[0], item[1]])
            # item[2] 是时间戳，我们将其转换为浮点数并保留
            temp_time.append([float(item[2]), float(0)]) # 第二个维度0可以是占位符或表示其他时间相关信息
        np_traj_coord.append(np.array(temp_coord))
        np_traj_time.append(np.array(temp_time))
    print(f"成功加载 {len(np_traj_coord)} 条轨迹数据。")

    start_t = time.time()
    if use_time_in_distance:
        print(f"开始使用 {distance_type} (带时间信息) 计算轨迹距离...")
        trajecotry_distance_list_time(np_traj_time, batch_size=batch_size, processors=20, distance_type=distance_type,
                                 data_name=data_type)
    else:
        print(f"开始使用 {distance_type} (不带时间信息) 计算轨迹距离...")
        trajecotry_distance_list(np_traj_coord, batch_size=batch_size, processors=20, distance_type=distance_type,
                             data_name=data_type)
    
    end_t = time.time()
    total = end_t - start_t
    print(f'距离计算完成。总耗时: {total:.2f} 秒。')


def main():
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description='Preprocess trajectory data and compute distances.')

    # 添加命令行参数
    parser.add_argument('--dataset', type=str, required=True, default= 'geolife', choices=['geolife', 'porto'],
                        help='Dataset to process (e.g., geolife or porto)')
    parser.add_argument('--distance', type=str, required=True, default = 'hausdorff',
                        choices=['hausdorff', 'dtw', 'lcss', 'erp'],
                        help='Type of distance metric (e.g., hausdorff, dtw, lcss, erp)')
    parser.add_argument('--num_traj', type=int, default=10000, # 设置一个默认值
                        help='Number of trajectories to process')
    parser.add_argument('--use_time_in_distance', action='store_true',
                        help='Use time information in distance computation (add this flag to enable)')

    # 解析命令行参数
    args = parser.parse_args()

    # 根据选择的数据集定义经纬度范围
    if args.dataset == 'geolife':
        lat_range = [39.6, 40.7]
        lon_range = [115.9, 117.1]
        data_path = './data/geolife/geolife' # 确保这个路径正确
        default_num_traj = 9000 # Geolife的默认轨迹数量
    elif args.dataset == 'porto':
        lat_range = [40.7, 41.8]
        lon_range = [-9.0, -7.9]
        data_path = './data/porto/porto' # 确保这个路径正确
        default_num_traj = 10000 # Porto的默认轨迹数量
    else:
        raise ValueError("Invalid dataset specified. Choose 'geolife' or 'porto'.")

    # 如果num_traj没有通过命令行指定，则使用数据集的默认值
    num_traj_to_process = args.num_traj if args.num_traj != 10000 else default_num_traj

    print(f"开始预处理 {args.dataset} 数据集...")
    # 生成轨迹特征文件
    coor_path, data_name_processed = trajectory_feature_generation(path=data_path,
                                                                    lat_range=lat_range,
                                                                    lon_range=lon_range,)
    print(f"轨迹特征已生成到: {coor_path}")

    # 调用距离计算函数
    distance_comp(coor_path, num_traj_to_process, args.distance, data_name_processed, args.use_time_in_distance)
    print("所有操作完成。")


if __name__ == '__main__':
    main()
