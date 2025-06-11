from tqdm import tqdm
from datetime import datetime
import csv
import pickle
import os

# --- 全局常量和配置 ---
# Porto数据集的经纬度范围
PORTO_LON_RANGE = [-9.0, -7.9]
PORTO_LAT_RANGE = [40.7, 41.8]

# --- 脚本主执行部分 ---
if __name__ == '__main__':
    print("脚本开始执行：Porto轨迹数据预处理")

    input_csv_path = 'D:\dataset\porto\train.csv'
    output_pickle_path = 'data/porto/porto'
    max_trajectories_to_process = 40000 # 对应原始代码中的限制
    
    try:
        csvFile = open(input_csv_path, 'r')
        reader = csv.reader(csvFile)
        print("成功打开输入CSV文件。")
    except FileNotFoundError:
        print(f"错误: 未找到CSV文件，请检查路径: {input_csv_path}")
        exit(1)
    except Exception as e:
        print(f"打开CSV文件时发生错误: {e}")
        exit(1)

    traj_missing = [] 
    trajectories = [] 
    
    min_lon, max_lon, min_lat, max_lat = -7.0, -10.0, 43.0, 40.0

    print("开始读取和初步解析CSV文件中的轨迹数据...")
    for item in tqdm(reader, desc="读取CSV"):
        if (reader.line_num == 1): # 跳过CSV文件头
            continue
        if (item[7] == 'True'):
            traj_missing.append(item[8]) # 记录缺失数据的轨迹ID
        elif (item[7] == 'False'):
            # item[8] 包含轨迹坐标字符串，需要去除前后的 '[[' 和 ']]'，然后按 '],[' 分割
            # item[5] 是开始时间戳 (EPOCH_TIME)
            # item[0] 是旅程ID
            trajectories.append((item[8][2:-2].split('],['), item[5], item[0]))

    csvFile.close()
    print(f"CSV文件读取完成。共识别到 {len(trajectories)} 条原始轨迹，{len(traj_missing)} 条缺失数据轨迹。")

    traj_porto = [] 
    processed_count = 0 
    
    print("开始解析、过滤轨迹点并进行范围检查...")
    for trajs in tqdm(trajectories, desc="处理轨迹"):
        # 检查是否达到最大处理轨迹数量限制
        if max_trajectories_to_process != -1 and processed_count >= max_trajectories_to_process:
            print(f"已达到最大处理轨迹数量限制 ({max_trajectories_to_process})，停止处理。")
            break

        # 确保轨迹点数量至少为10，过短的轨迹可能没有研究价值
        if (len(trajs[0]) >= 10):
            Traj = [] 
            inrange = True 
            
            tmp_max_lon = max_lon
            tmp_min_lat = min_lat
            tmp_max_lat = max_lat
            
            point_idx = 0 
            
            for point_str in trajs[0]:
                tr = point_str.split(',')
                
                # 确保经纬度数据不为空
                if (tr[0] != '' and tr[1] != ''):
                    lon = float(tr[0])
                    lat = float(tr[1])
                    
                    # 检查轨迹点是否在定义的Porto经纬度范围之外
                    if ((lat < PORTO_LAT_RANGE[0]) or (lat > PORTO_LAT_RANGE[1]) or 
                        (lon < PORTO_LON_RANGE[0]) or (lon > PORTO_LON_RANGE[1])):
                        inrange = False 
                        break
                    
                    # 更新当前轨迹的局部经纬度范围
                    tmp_min_lon = min(tmp_min_lon, lon)
                    tmp_max_lon = max(tmp_max_lon, lon)
                    tmp_min_lat = min(tmp_min_lat, lat)
                    tmp_max_lat = max(tmp_max_lat, lat)
                    
                    # 计算轨迹点的时间戳和datetime对象
                    point_timestamp = int(trajs[-2]) + point_idx * 15
                    start_dtime = datetime.fromtimestamp(point_timestamp)
                    
                    # 将解析后的点数据添加到当前轨迹列表
                    Traj.append((lon, lat, point_timestamp, start_dtime))
                    point_idx += 1 # 轨迹点索引递增
            
            # 如果整个轨迹都在有效范围内且轨迹点数量符合要求
            if inrange != False:
                traj_porto.append(Traj)
                processed_count += 1
                
                min_lon = tmp_min_lon
                max_lon = tmp_max_lon
                min_lat = tmp_min_lat
                max_lat = tmp_max_lat

    print(f"所有轨迹处理完毕。最终筛选并保存了 {len(traj_porto)} 条有效轨迹。")
    print(f"最终经度范围: [{min_lon:.4f}, {max_lon:.4f}]")
    print(f"最终纬度范围: [{min_lat:.4f}, {max_lat:.4f}]")

    # 将处理后的轨迹数据序列化并保存到文件中
    output_dir = os.path.dirname(output_pickle_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"正在将处理后的轨迹数据保存到文件: {output_pickle_path}")
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(traj_porto, f)
    print("数据保存完成。脚本执行结束。")