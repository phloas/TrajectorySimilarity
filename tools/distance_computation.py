"""
distance_computation.py

该模块提供了计算轨迹之间距离的功能，支持多种距离度量方法（如Hausdorff, DTW, LCSS, EDR, ERP）。
它利用多进程并行计算，以提高大规模轨迹数据处理的效率。

主要功能包括：
-   `trajectory_distance`：根据轨迹特征映射计算批量空间距离（旧版函数，可能不再主要使用）。
-   `trajecotry_distance_list`：批量计算轨迹列表之间的空间距离。
-   `trajectory_distance_batch`：执行单个批次的轨迹空间距离计算，并将结果保存。
-   `trajectory_distance_combine`：合并所有批次计算的空间距离结果。
-   `trajecotry_distance_list_time`：批量计算轨迹列表之间的时间距离（考虑时间信息）。
-   `trajectory_distance_batch_time`：执行单个批次的轨迹时间距离计算，并将结果保存。
-   `trajectory_distance_combine_time`：合并所有批次计算的时间距离结果。
"""

import traj_dist.distance as tdist
import os
import numpy as np
import multiprocessing
import tqdm
import os.path as osp
import config

# --- 空间距离计算函数 ---

def trajectory_distance(traj_feature_map, traj_keys,  distance_type=config.distance_type, batch_size=config.batch_size, processors=30):
    """
    根据轨迹特征映射计算批量轨迹之间的空间距离（此函数可能为旧版或特定用途）。

    参数:
    traj_feature_map (dict): 轨迹ID到轨迹点列表的映射。
    traj_keys (list): 要处理的轨迹ID列表。
    distance_type (str, optional): 距离度量类型。默认为 "hausdorff"。
    batch_size (int, optional): 每个批次处理的轨迹数量。默认为50。
    processors (int, optional): 用于并行计算的进程数量。默认为30。
    """
    print(f"(旧版) 开始计算批量空间距离，距离类型: {distance_type}")
    trajs = []
    for k in traj_keys:
        traj = []
        for record in traj_feature_map[k]:
            # 假设 record[1] 是纬度，record[2] 是经度 (根据 generate_feature.py 中的 new_traj.append([0, p[1], p[0]...]) 推断)
            traj.append([record[1], record[2]]) # 提取 (纬度, 经度) 格式
        trajs.append(np.array(traj))

    # 创建进程池
    pool = multiprocessing.Pool(processes=processors)
    batch_number = 0
    # 遍历轨迹并按批次提交任务给进程池
    for i in range(len(trajs)):
        if (i != 0) and (i % batch_size == 0): # 当i是batch_size的倍数时，提交一个批次
            start_idx = batch_size * batch_number
            end_idx = i
            print(f"提交空间距离计算批次: {start_idx}-{end_idx}")
            # 使用 apply_async 异步执行批处理任务
            pool.apply_async(trajectory_distance_batch, (end_idx, trajs[start_idx:end_idx], trajs, distance_type,
                                                         'geolife')) # 硬编码 'geolife'，可能需要根据实际情况修改
            batch_number += 1
    
    # 处理剩余的轨迹（如果总数不是batch_size的整数倍）
    if len(trajs) % batch_size != 0:
        start_idx = batch_size * batch_number
        end_idx = len(trajs)
        if start_idx < end_idx: # 确保有剩余轨迹
            print(f"提交空间距离计算最后一个批次: {start_idx}-{end_idx}")
            pool.apply_async(trajectory_distance_batch, (end_idx, trajs[start_idx:end_idx], trajs, distance_type,
                                                         'geolife'))

    pool.close() # 关闭进程池，不再接受新的任务
    pool.join() # 等待所有子进程完成
    print("(旧版) 所有空间距离批次计算完成并保存。")


def trajecotry_distance_list(trajs, distance_type=config.distance_type, batch_size=config.batch_size, processors=30, data_name=config.data_type):
    """
    批量计算轨迹列表之间的空间距离，并保存中间结果。此函数是主要的批量空间距离计算入口。

    参数:
    trajs (list): 轨迹数据列表，每个元素是一个NumPy数组，包含轨迹点的经纬度坐标。
    distance (str, optional): 距离度量类型。默认为 "hausdorff"。
    batch_size (int, optional): 每个批次处理的轨迹数量。默认为50。
    processors (int, optional): 用于并行计算的进程数量。默认为30。
    data_name (str, optional): 数据集名称（例如 'porto' 或 'geolife'）。默认为 'porto'。
    """
    print(f"开始为数据集 '{data_name}' 计算空间距离，距离类型: {distance_type}")
    pool = multiprocessing.Pool(processes=processors) # 创建进程池
    
    # 遍历轨迹并按批次提交任务给进程池
    # range(len(trajs)+1) 确保能覆盖到最后一个批次，即使它不足batch_size
    for i in range(len(trajs) + 1):
        if (i != 0) and (i % batch_size == 0): # 当i是batch_size的倍数时，提交一个批次
            start_idx = i - batch_size
            end_idx = i
            print(f"提交空间距离计算批次: {start_idx}-{end_idx}")
            pool.apply_async(trajectory_distance_batch, (end_idx, trajs[start_idx:end_idx], trajs, distance_type,
                                                         data_name))
    
    # 处理可能存在的最后一个不足batch_size的批次
    if len(trajs) % batch_size != 0:
        start_idx = (len(trajs) // batch_size) * batch_size
        end_idx = len(trajs)
        if start_idx < end_idx:
            print(f"提交空间距离计算最后一个批次: {start_idx}-{end_idx}")
            pool.apply_async(trajectory_distance_batch, (end_idx, trajs[start_idx:end_idx], trajs, distance_type,
                                                         data_name))

    pool.close()
    pool.join()
    print(f"数据集 '{data_name}' 的所有空间距离批次计算完成。")


def trajectory_distance_batch(i, batch_trjs, trjs, distance_type=config.distance_type, data_name=config.data_type):
    """
    计算一个批次轨迹与所有轨迹之间的空间距离矩阵，并保存结果。

    参数:
    i (int): 当前批次的结束索引，用于命名保存的文件。
    batch_trjs (list): 当前批次的轨迹列表，每个元素是一个NumPy数组。
    trjs (list): 所有轨迹的列表，每个元素是一个NumPy数组。
    metric_type (str, optional): 距离度量类型。默认为 "hausdorff"。
    data_name (str, optional): 数据集名称。默认为 'porto'。
    """
    print(f"进程 {multiprocessing.current_process().name} 正在计算批次 {i} 的空间距离...")
    trs_matrix = None

    if distance_type == 'lcss':
        # LCSS (Longest Common Subsequence) 距离计算，eps为阈值
        # tdist.cdist 返回的是相似度 (0到1)，需要转换为距离
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=distance_type, eps=0.003)
        tmp_matrix = 1.0 - trs_matrix # 相似度转为不相似度
        len_a = len(batch_trjs)
        len_b = len(trjs)
        min_len_matrix = np.ones((len_a, len_b)) # 最小长度矩阵
        sum_len_matrix = np.ones((len_a, len_b)) # 长度之和矩阵
        for ii in range(len_a):
            for jj in range(len_b):
                min_len_matrix[ii][jj] = min(len(batch_trjs[ii]), len(trjs[jj]))
                sum_len_matrix[ii][jj] = len(batch_trjs[ii]) + len(trjs[jj])
        tmp_trs_matrix = tmp_matrix * min_len_matrix # 乘以最小长度
        trs_matrix = sum_len_matrix - 2.0 * tmp_trs_matrix # 最终LCSS距离公式
    elif distance_type == 'edr':
        # EDR (Edit Distance on Real sequence) 距离计算
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=distance_type, eps=0.003)
        len_a = len(batch_trjs)
        len_b = len(trjs)
        max_len_matrix = np.ones((len_a, len_b))
        for ii in range(len_a):
            for jj in range(len_b):
                max_len_matrix[ii][jj] = max(len(batch_trjs[ii]), len(trjs[jj]))
        trs_matrix = trs_matrix * max_len_matrix # EDR距离公式
    elif distance_type == 'erp':
        # ERP (Edit Distance with Real Penalty) 距离计算
        # aa 是一个参考点，用于计算删除/插入点的惩罚。这里硬编码了Porto范围的中心点。
        aa = np.zeros(2, dtype=float)
        aa[0] = 40.0  # Porto数据的纬度参考点 (接近范围中心)
        aa[1] = -10.0 # Porto数据的经度参考点 (接近范围中心)
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=distance_type, g=aa)
    else:
        # 对于其他距离类型 (如hausdorff, dtw)，直接使用cdist
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=distance_type)

    trs_matrix = np.array(trs_matrix) # 转换为NumPy数组
    
    # 构建保存路径
    output_dir = './ground_truth/{}/{}/{}_spatial_batch/'.format(data_name, str(distance_type), str(distance_type))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) # 如果目录不存在则创建
    
    # 保存批次结果为 .npy 文件
    output_file_path = osp.join(output_dir, f'{i}_spatial_distance.npy')
    np.save(output_file_path, trs_matrix)
    print(f'[PROGRESS] 空间距离批次 {i} 计算并保存完成: {output_file_path}')

def trajectory_distance_combine(trajs_len, batch_size, metric_type, data_name):
    """
    合并所有批次计算的空间距离结果，生成最终的距离矩阵文件。

    参数:
    trajs_len (int): 原始轨迹的总数量。
    batch_size (int): 每个批次处理的轨迹数量。
    metric_type (str): 距离度量类型。
    data_name (str): 数据集名称。
    """
    print(f"[INFO] 开始合并数据集 '{data_name}' 的 {metric_type} 空间距离批次文件...")
    res = [] # 存储加载的批次距离矩阵
    # 遍历所有可能的批次文件
    for i in tqdm(range(trajs_len + 1), desc="合并空间距离批次"):
        if i != 0 and i % batch_size == 0:
            batch_file_path = './ground_truth/{}/{}/{}_spatial_batch/{}_spatial_distance.npy'.format(data_name,
                                                                                                      str(metric_type),
                                                                                                      str(metric_type),
                                                                                                      str(i))
            try:
                res.append(np.load(batch_file_path))
            except FileNotFoundError:
                print(f"[WARNING] 未找到批次文件: {batch_file_path}。跳过。")
            except Exception as e:
                print(f"[ERROR] 加载批次文件 {batch_file_path} 时发生错误: {e}")
    
    # 处理最后一个可能不完整的批次文件
    if trajs_len % batch_size != 0:
        last_batch_i = (trajs_len // batch_size + 1) * batch_size # 最后一个批次的文件名索引
        batch_file_path = './ground_truth/{}/{}/{}_spatial_batch/{}_spatial_distance.npy'.format(data_name,
                                                                                                str(metric_type),
                                                                                                str(metric_type),
                                                                                                str(last_batch_i))
        try:
            if os.path.exists(batch_file_path): # 检查文件是否存在再尝试加载
                res.append(np.load(batch_file_path))
            else:
                print(f"未找到最后一个不完整批次的文件: {batch_file_path}。这可能是正常的。")
        except Exception as e:
            print(f"加载最后一个批次文件 {batch_file_path} 时发生错误: {e}")

    if not res:
        print("没有找到任何空间距离批次文件可合并。")
        return

    # 垂直堆叠所有批次矩阵
    res = np.concatenate(res, axis=0)
    
    # 保存最终合并的距离矩阵
    final_output_path = './ground_truth/{}/{}/{}_spatial_distance.npy'.format(data_name, str(metric_type), str(metric_type))
    np.save(final_output_path, res)
    print(f'[INFO] 成功合并空间距离地面真值。结果保存到: {final_output_path}')

# --- 时间距离计算函数 (考虑时间信息) ---

def trajecotry_distance_list_time(trajs, distance_type=config.distance_type, batch_size=config.batch_size, processors=30, data_name=config.data_type):
    """
    批量计算轨迹列表之间的时间距离（考虑时间信息），并保存中间结果。此函数是主要的批量时间距离计算入口。

    参数:
    trajs (list): 轨迹数据列表，每个元素是一个NumPy数组，包含轨迹点的时间戳信息。
    distance_type (str, optional): 距离度量类型。默认为 "hausdorff"。
    batch_size (int, optional): 每个批次处理的轨迹数量。默认为50。
    processors (int, optional): 用于并行计算的进程数量。默认为30。
    data_name (str, optional): 数据集名称。默认为 'porto'。
    """
    print(f"[INFO] 开始为数据集 '{data_name}' 计算时间距离，距离类型: {distance_type}")
    pool = multiprocessing.Pool(processes=processors) # 创建进程池
    
    # 遍历轨迹并按批次提交任务给进程池
    for i in range(len(trajs) + 1):
        if (i != 0) and (i % batch_size == 0):
            start_idx = i - batch_size
            end_idx = i
            print(f"提交时间距离计算批次: {start_idx}-{end_idx}")
            pool.apply_async(trajectory_distance_batch_time, (end_idx, trajs[start_idx:end_idx], trajs, distance_type,
                                                              data_name))
    
    # 处理可能存在的最后一个不足batch_size的批次
    if len(trajs) % batch_size != 0:
        start_idx = (len(trajs) // batch_size) * batch_size
        end_idx = len(trajs)
        if start_idx < end_idx:
            print(f"提交时间距离计算最后一个批次: {start_idx}-{end_idx}")
            pool.apply_async(trajectory_distance_batch_time, (end_idx, trajs[start_idx:end_idx], trajs, distance_type,
                                                              data_name))

    pool.close() # 关闭进程池
    pool.join() # 等待所有子进程完成
    print(f"数据集 '{data_name}' 的所有时间距离批次计算完成。")


def trajectory_distance_batch_time(i, batch_trjs, trjs, metric_type="hausdorff", data_name='porto'):
    """
    计算一个批次轨迹与所有轨迹之间的时间距离矩阵，并保存结果。
    此函数与 `trajectory_distance_batch` 类似，但针对包含时间信息的轨迹数据进行操作。

    参数:
    i (int): 当前批次的结束索引，用于命名保存的文件。
    batch_trjs (list): 当前批次的轨迹列表，每个元素是一个NumPy数组，包含时间戳信息。
    trjs (list): 所有轨迹的列表，每个元素是一个NumPy数组，包含时间戳信息。
    metric_type (str, optional): 距离度量类型。默认为 "hausdorff"。
    data_name (str, optional): 数据集名称。默认为 'porto'。
    """
    print(f"进程 {multiprocessing.current_process().name} 正在计算批次 {i} 的时间距离...")
    trs_matrix = None

    if metric_type == 'lcss':
        # LCSS (Longest Common Subsequence) 距离计算，eps为阈值
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type, eps=0.003)
        tmp_matrix = 1.0 - trs_matrix
        len_a = len(batch_trjs)
        len_b = len(trjs)
        min_len_matrix = np.ones((len_a, len_b))
        sum_len_matrix = np.ones((len_a, len_b))
        for ii in range(len_a):
            for jj in range(len_b):
                min_len_matrix[ii][jj] = min(len(batch_trjs[ii]), len(trjs[jj]))
                sum_len_matrix[ii][jj] = len(batch_trjs[ii]) + len(trjs[jj])
        tmp_trs_matrix = tmp_matrix * min_len_matrix
        trs_matrix = sum_len_matrix - 2.0 * tmp_trs_matrix
    elif metric_type == 'edr':
        # EDR (Edit Distance on Real sequence) 距离计算
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type, eps=0.003)
        len_a = len(batch_trjs)
        len_b = len(trjs)
        max_len_matrix = np.ones((len_a, len_b))
        for ii in range(len_a):
            for jj in range(len_b):
                max_len_matrix[ii][jj] = max(len(batch_trjs[ii]), len(trjs[jj]))
        trs_matrix = trs_matrix * max_len_matrix
    elif metric_type == 'erp':
        # ERP (Edit Distance with Real Penalty) 距离计算
        # aa 在时间距离中通常表示时间上的"空"点或惩罚参数，这里硬编码为较大的值和0
        aa = np.zeros(2, dtype=float)
        aa[0] = 1000000000  # 时间维度上的一个大惩罚值
        aa[1] = 0           # 另一个维度（可能未使用或作为占位符）
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type, g=aa)
    else:
        # 对于其他距离类型，直接使用cdist
        trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type)

    trs_matrix = np.array(trs_matrix)
    
    # 构建保存路径
    output_dir = './ground_truth/{}/{}/{}_temporal_batch/'.format(data_name, str(metric_type), str(metric_type))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) # 如果目录不存在则创建
    
    # 保存批次结果为 .npy 文件
    output_file_path = osp.join(output_dir, f'{i}_temporal_distance.npy')
    np.save(output_file_path, trs_matrix)
    print(f'[PROGRESS] 时间距离批次 {i} 计算并保存完成: {output_file_path}')


def trajectory_distance_combine_time(trajs_len, batch_size, metric_type, data_name):
    """
    合并所有批次计算的时间距离结果，生成最终的距离矩阵文件。

    参数:
    trajs_len (int): 原始轨迹的总数量。
    batch_size (int): 每个批次处理的轨迹数量。
    metric_type (str): 距离度量类型。
    data_name (str): 数据集名称。
    """
    print(f"[INFO] 开始合并数据集 '{data_name}' 的 {metric_type} 时间距离批次文件...")
    res = [] # 存储加载的批次距离矩阵
    # 遍历所有可能的批次文件
    for i in tqdm(range(trajs_len + 1), desc="合并时间距离批次"):
        if i != 0 and i % batch_size == 0:
            batch_file_path = './ground_truth/{}/{}/{}_temporal_batch/{}_temporal_distance.npy'.format(data_name,
                                                                                                        str(metric_type),
                                                                                                        str(metric_type),
                                                                                                        str(i))
            try:
                res.append(np.load(batch_file_path))
            except FileNotFoundError:
                print(f"[WARNING] 未找到批次文件: {batch_file_path}。跳过。")
            except Exception as e:
                print(f"[ERROR] 加载批次文件 {batch_file_path} 时发生错误: {e}")

    # 处理最后一个可能不完整的批次文件
    if trajs_len % batch_size != 0:
        last_batch_i = (trajs_len // batch_size + 1) * batch_size # 最后一个批次的文件名索引
        batch_file_path = './ground_truth/{}/{}/{}_temporal_batch/{}_temporal_distance.npy'.format(data_name,
                                                                                                str(metric_type),
                                                                                                str(metric_type),
                                                                                                str(last_batch_i))
        try:
            if os.path.exists(batch_file_path): # 检查文件是否存在再尝试加载
                res.append(np.load(batch_file_path))
            else:
                print(f"[WARNING] 未找到最后一个不完整批次的文件: {batch_file_path}。这可能是正常的。")
        except Exception as e:
            print(f"[ERROR] 加载最后一个批次文件 {batch_file_path} 时发生错误: {e}")

    if not res:
        print("[ERROR] 没有找到任何时间距离批次文件可合并。")
        return

    # 垂直堆叠所有批次矩阵
    res = np.concatenate(res, axis=0)
    
    # 保存最终合并的距离矩阵
    final_output_path = './ground_truth/{}/{}/{}_temporal_distance.npy'.format(data_name, str(metric_type), str(metric_type))
    np.save(final_output_path, res)
    print(f'成功合并时间距离地面真值。结果保存到: {final_output_path}')
