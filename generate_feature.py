"""
generate_feature.py

该脚本包含了用于轨迹数据预处理和特征生成的类和函数，主要功能包括：
1.  将原始轨迹数据（经纬度，时间戳等）进行网格化处理。
2.  为每个唯一的网格单元或坐标点分配唯一ID。
3.  生成简化后的网格轨迹序列和坐标轨迹序列。
4.  根据指定的数据集（Geolife或Porto）和范围对轨迹进行过滤。
5.  将生成的轨迹特征（原始轨迹索引、网格化后的坐标和网格序列）保存为pickle文件。
"""

import pickle
import os.path as osp
from tqdm import tqdm

class Preprocesser(object):
    """
    Preprocesser类用于将轨迹数据进行网格化处理，并生成简化后的轨迹序列。
    它根据设定的网格大小 (delta) 和地理范围 (lat_range, lon_range) 来计算轨迹点所在的网格索引。
    """
    def __init__(self, delta=0.005, lat_range=[1, 2], lon_range=[1, 2]):
        """
        初始化Preprocesser对象。

        参数:
        delta (float, optional): 网格单元的大小 (经纬度)。默认为0.005。
        lat_range (list, optional): 纬度范围 [min_lat, max_lat]。默认为 [1, 2]。
        lon_range (list, optional): 经度范围 [min_lon, max_lon]。默认为 [1, 2]。
        """
        print(f"Preprocesser初始化：delta={delta}, 纬度范围={lat_range}, 经度范围={lon_range}")
        self.delta = delta
        self.lat_range = lat_range
        self.lon_range = lon_range
        self._init_grid_hash_function()

    def _init_grid_hash_function(self):
        """
        初始化网格哈希函数所需的经纬度网格列表 (self.x, self.y)。
        这些列表定义了每个网格单元的边界。
        """
        print("正在初始化网格哈希函数...")
        dXMax, dXMin = self.lon_range[1], self.lon_range[0]
        dYMax, dYMin = self.lat_range[1], self.lat_range[0]
        
        # 生成经度网格边界
        self.x = self._frange(dXMin, dXMax, self.delta)
        # 生成纬度网格边界
        self.y = self._frange(dYMin, dYMax, self.delta)
        print(f"网格初始化完成。经度网格数量: {len(self.x)}, 纬度网格数量: {len(self.y)}")

    def _frange(self, start, end=None, inc=None):
        """
        一个支持浮点数增量的范围函数，类似于内置的range()。

        参数:
        start (float): 起始值。
        end (float, optional): 结束值 (不包含)。如果为None，则start是结束值，起始值为0.0。
        inc (float, optional): 增量。默认为1.0。

        返回:
        list: 包含浮点数的列表。
        """
        if end is None:
            end = start + 0.0
            start = 0.0
        if inc is None:
            inc = 1.0
        L = []
        i = 0
        while True:
            next_val = start + i * inc
            # 根据增量方向判断是否超出结束值
            if (inc > 0 and next_val >= end) or (inc < 0 and next_val <= end):
                break
            L.append(next_val)
            i += 1
        return L

    def get_grid_index(self, point_tuple):
        """
        根据经纬度坐标点计算其在网格中的索引。

        参数:
        point_tuple (tuple): 包含经度 (lon) 和纬度 (lat) 的元组，例如 (lon, lat)。

        返回:
        tuple: 包含X轴网格索引、Y轴网格索引和一维网格索引的元组 (x_grid, y_grid, index)。
        """
        test_x, test_y = point_tuple[0], point_tuple[1]
        
        # 计算X轴（经度）上的网格索引
        x_grid = int((test_x - self.lon_range[0]) / self.delta)
        # 计算Y轴（纬度）上的网格索引
        y_grid = int((test_y - self.lat_range[0]) / self.delta)
        
        # 计算一维网格索引：(y_grid * 经度网格总数) + x_grid
        index = (y_grid) * (len(self.x)) + x_grid
        return x_grid, y_grid, index

    def traj2grid_seq(self, trajs_points, isCoordinate=False):
        """
        将原始轨迹点序列转换为网格轨迹序列，并去除重复的相邻网格点。

        参数:
        trajs_points (list): 包含轨迹点的列表。每个点可能包含 [占位符, 纬度, 经度, 时间戳, datetime对象]。
                             注意：这里经纬度是 (r[2], r[1]) 的顺序，即 (经度, 纬度)。
        isCoordinate (bool, optional): 如果为True，则保留原始坐标信息而不是网格索引。默认为False。

        返回:
        list: 简化后的网格ID序列或坐标序列。
        """
        grid_traj = []
        # 将每个轨迹点转换为其对应的网格索引
        for r in trajs_points:
            x_grid, y_grid, index = self.get_grid_index((r[2], r[1]))
            grid_traj.append(index)
        
        privious = None 
        hash_traj = []
        
        # 遍历网格轨迹序列，去除相邻重复的网格ID
        for index, i in enumerate(grid_traj):
            if privious is None:
                privious = i
                if not isCoordinate:
                    hash_traj.append(i)
                else:
                    hash_traj.append(trajs_points[index][1:]) # 如果isCoordinate为True，保留原始坐标 (从索引1开始)
            else:
                if i == privious: # 如果当前网格ID与前一个相同，则跳过
                    pass
                else: # 否则，添加当前网格ID并更新前一个网格ID
                    if not isCoordinate:
                        hash_traj.append(i)
                    else:
                        hash_traj.append(trajs_points[index][1:])
                    privious = i
        return hash_traj

    def _traj2grid_preprocess(self, traj_feature_map, isCoordinate=False):
        """
        对轨迹特征映射中的所有轨迹进行网格化预处理。

        参数:
        traj_feature_map (dict): 轨迹ID到原始轨迹点列表的映射。
        isCoordinate (bool, optional): 是否保留原始坐标信息。默认为False。

        返回:
        list: 包含所有简化后网格轨迹序列的列表。
        """
        trajs_hash = []
        trajs_keys = traj_feature_map.keys()
        print(f"正在将 {len(trajs_keys)} 条轨迹转换为网格/坐标序列...")
        for traj_key in tqdm(trajs_keys, desc="网格化轨迹"):
            traj = traj_feature_map[traj_key]
            trajs_hash.append(self.traj2grid_seq(traj, isCoordinate)) # 调用 traj2grid_seq 进行处理
        return trajs_hash

    def preprocess(self, traj_feature_map, isCoordinate=False):
        """
        对轨迹数据进行全面的预处理，包括网格化、去重和统计有用网格。

        参数:
        traj_feature_map (dict): 轨迹ID到原始轨迹点列表的映射。
        isCoordinate (bool, optional): 如果为True，则生成坐标序列而不是网格ID序列。默认为False。

        返回:
        tuple: 包含 (处理后的轨迹列表, 有用网格字典, 最大轨迹长度)。
               useful_grids: 仅在 isCoordinate=False 时有效，存储网格ID及其出现次数。
        """
        if not isCoordinate:
            traj_grids = self._traj2grid_preprocess(traj_feature_map)
            print(f'[INFO] 原始网格轨迹数量: {len(traj_grids)}')

            useful_grids = {}
            total_grid_points = 0
            max_len = 0 # 记录最长网格轨迹的长度

            print("正在统计网格点和轨迹长度...")
            for i, traj in enumerate(traj_grids):
                if len(traj) > max_len:
                    max_len = len(traj)
                total_grid_points += len(traj)
                for grid in traj:
                    # 统计每个网格ID的出现次数
                    if grid in useful_grids:
                        useful_grids[grid][1] += 1
                    else:
                        useful_grids[grid] = [len(useful_grids) + 1, 1] # [唯一ID, 计数]
            print(f"唯一网格数量: {len(useful_grids.keys())}")
            print(f"总网格点数: {total_grid_points}, 最长网格轨迹长度: {max_len}")
            return traj_grids, useful_grids, max_len
        elif isCoordinate:
            # 将原始轨迹转换为坐标序列 (不进行网格去重)
            traj_coords_processed = self._traj2grid_preprocess(traj_feature_map, isCoordinate=isCoordinate)
            max_len = 0 # 记录最长坐标轨迹的长度
            useful_grids = {} 
            
            print("正在计算坐标轨迹的最大长度...")
            for i, traj in enumerate(traj_coords_processed):
                if len(traj) > max_len:
                    max_len = len(traj)
            print(f"最长坐标轨迹长度: {max_len}")
            return traj_coords_processed, useful_grids, max_len


def trajectory_feature_generation(path=None,
                                  lat_range=None,
                                  lon_range=None,
                                  min_length=50):
    """
    从原始数据文件中加载轨迹，进行过滤和特征生成（网格化、坐标序列）。

    参数:
    path (str, optional): 原始轨迹数据文件的路径 (pickle文件)。例如 './data/porto/porto'。
    lat_range (list, optional): 纬度范围 [min_lat, max_lat]，用于过滤轨迹点。默认为porto_lat_range。
    lon_range (list, optional): 经度范围 [min_lon, max_lon]，用于过滤轨迹点。默认为porto_lon_range。
    min_length (int, optional): 轨迹的最小长度（点数），用于过滤过短的轨迹。默认为50。

    返回:
    tuple: 包含 (轨迹坐标特征文件路径, 数据集名称)。
           生成的特征文件包括：
           - *_traj_index: 筛选后的原始轨迹字典
           - *_traj_coord: 网格化或坐标化后的轨迹坐标序列
           - *_traj_grid: 相对网格位置的X,Y坐标序列
    """
    if path is None:
        print("轨迹数据文件路径 'path' 不能为空。")
        return None, None

    # 从文件路径中提取数据集名称 (例如 'porto' 或 'geolife')
    fname = path.split('/')[-1].split('_')[0] 
    print(f"开始为数据集 '{fname}' 生成轨迹特征。")
    print(f"输入数据文件: {path}")
    print(f"地理过滤范围: 纬度 {lat_range}, 经度 {lon_range}")
    print(f"最小轨迹长度限制: {min_length}")

    # 加载原始轨迹数据
    try:
        with open(path, 'rb') as f:
            trajs_raw = pickle.load(f)
        print(f"成功加载 {len(trajs_raw)} 条原始轨迹数据。")
    except FileNotFoundError:
        print(f"原始轨迹数据文件未找到: {path}。请确保文件存在。")
        return None, None
    except Exception as e:
        print(f"加载原始轨迹数据时发生错误: {e}")
        return None, None

    traj_index = {} # 存储筛选后的原始轨迹，键为原始索引，值为轨迹点列表
    max_len_raw_traj = 0 # 记录筛选后原始轨迹的最大长度

    # 初始化Preprocesser，用于将轨迹点映射到网格
    # delta=0.001 是一个经验值，用于网格划分的粒度
    preprocessor = Preprocesser(delta=0.001, lat_range=lat_range, lon_range=lon_range)
    
    # 打印网格的最大索引，用于验证网格化是否正确
    # 注意: get_grid_index 期望 (lon, lat) 顺序，这里是 (lon_range[1], lat_range[1]) 即最大经纬度
    _, _, max_grid_index = preprocessor.get_grid_index((lon_range[1], lat_range[1]))
    print(f"网格最大索引示例 (lon_max, lat_max): {max_grid_index}")

    print("开始遍历并过滤原始轨迹...")
    valid_traj_count = 0 # 记录有效轨迹的数量
    for i, traj in enumerate(tqdm(trajs_raw, desc="过滤轨迹")):
        new_traj = [] # 临时列表，用于存储当前轨迹的解析和扩充后的点数据
        
        # 过滤轨迹：长度必须大于等于 min_length
        if (len(traj) >= min_length):
            is_current_traj_in_range = True # 标记当前轨迹是否完全在指定地理范围内
            for p in traj:
                lon, lat = p[0], p[1] # p[0] 是经度，p[1] 是纬度
                
                # 检查轨迹点是否在指定的经纬度范围之外
                if not ((lat > lat_range[0]) and (lat < lat_range[1]) and 
                        (lon > lon_range[0]) and (lon < lon_range[1])):
                    is_current_traj_in_range = False # 标记为超出范围
                    break 
                
                # 原始轨迹点格式：p[0]=lon, p[1]=lat, p[2]=timestamp, p[3]=datetime_object
                # new_traj 格式：[0, lat, lon, timestamp, datetime_object]
                new_traj.append([0, p[1], p[0], p[2], p[3]])
                # new_traj_record.append([p[0], p[1]]) # 存储原始经纬度 (lon, lat)

            if is_current_traj_in_range: # 如果整个轨迹都在有效范围内
                coor_traj = preprocessor.traj2grid_seq(new_traj, isCoordinate=True)
                
                # 进一步过滤：坐标序列长度必须在 [10, 150) 之间
                if ((len(coor_traj) > 10) and (len(coor_traj) < 150)):
                    if len(traj) > max_len_raw_traj:
                        max_len_raw_traj = len(traj)
                    traj_index[i] = new_traj
                    valid_traj_count += 1

        if i % 2000 == 0: # 每处理2000条轨迹输出一次进度
            print(f"当前处理至原始轨迹 {i}, 已筛选出 {valid_traj_count} 条符合条件的轨迹。")

    # 打印最终的统计信息
    print(f"原始轨迹过滤完成。最终筛选出 {len(traj_index.keys())} 条有效轨迹。")
    print(f"筛选后原始轨迹的最大长度: {max_len_raw_traj}")

    # --- 步骤1: 保存筛选后的原始轨迹索引 ---
    output_traj_index_path = osp.join('features', f'{fname}_traj_index')
    print(f"正在保存筛选后的原始轨迹索引到: {output_traj_index_path}")
    try:
        with open(output_traj_index_path, 'wb') as f:
            pickle.dump(traj_index, f)  # 筛选完之后的原始轨迹 (例如 beijing:9553, porto:620876)
        print("轨迹索引保存成功。")
    except Exception as e:
        print(f"保存轨迹索引时发生错误: {e}")

    # --- 步骤2: 对筛选后的轨迹进行预处理，生成坐标序列 ---
    print("正在对筛选后的轨迹进行预处理，生成坐标序列...")
    # isCoordinate=True 确保返回的是坐标序列，而非网格ID序列
    trajs_processed_coord, useful_grids_dummy, max_len_processed_coord = preprocessor.preprocess(traj_index, isCoordinate=True)
    
    print(f"坐标序列生成完成。第一条坐标序列示例: {trajs_processed_coord[0] if trajs_processed_coord else 'N/A'}")

    # 保存坐标序列数据
    output_traj_coord_path = osp.join('features', f'{fname}_traj_coord')
    print(f"正在保存轨迹坐标序列到: {output_traj_coord_path}")
    try:
        with open(output_traj_coord_path, 'wb') as f:
            pickle.dump((trajs_processed_coord, max_len_processed_coord), f) # 包含筛选后映射成网格后对应的经纬度
        print("轨迹坐标序列保存成功。")
    except Exception as e:
        print(f"保存轨迹坐标序列时发生错误: {e}")

    # --- 步骤3: 生成相对网格位置的X, Y坐标序列 ---
    all_trajs_grids_xy = []
    # 初始化网格X, Y坐标的边界，用于计算相对位置
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    print("正在计算网格坐标的边界...")
    for traj_coord_seq in tqdm(trajs_processed_coord, desc="计算网格边界"):
        for point in traj_coord_seq:
            # point[0] 是经度 (lon), point[1] 是纬度 (lat)
            x, y, _ = preprocessor.get_grid_index((point[0], point[1]))
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
    print(f"网格行列的边界 (X: {min_x}-{max_x}, Y: {min_y}-{max_y})")

    print("正在生成相对网格位置的X, Y坐标序列...")
    for traj_coord_seq in tqdm(trajs_processed_coord, desc="生成相对网格"): 
        traj_grid_xy = []
        for point in traj_coord_seq:
            x, y, _ = preprocessor.get_grid_index((point[0], point[1]))
            x_relative = x - min_x
            y_relative = y - min_y
            grids_xy = [y_relative, x_relative] # 存储为 [y_relative, x_relative] 形式
            traj_grid_xy.append(grids_xy)
        all_trajs_grids_xy.append(traj_grid_xy)
    
    print(f"相对网格坐标序列生成完成。第一条示例: {all_trajs_grids_xy[0] if all_trajs_grids_xy else 'N/A'}")
    print(f"总共生成 {len(all_trajs_grids_xy)} 条相对网格坐标序列。")

    # 保存相对网格坐标序列数据
    output_traj_grid_path = osp.join('features', f'{fname}_traj_grid')
    print(f"正在保存相对网格坐标序列到: {output_traj_grid_path}")
    try:
        with open(output_traj_grid_path, 'wb') as f:
            pickle.dump((all_trajs_grids_xy, max_len_processed_coord), f) # 记录相对网格位置的x，y
        print("相对网格坐标序列保存成功。")
    except Exception as e:
        print(f"保存相对网格坐标序列时发生错误: {e}")

    print(f"数据集 '{fname}' 的特征生成流程完成。")
    return output_traj_coord_path, fname
