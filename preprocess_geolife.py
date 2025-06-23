import os
import numpy as np
import pickle
import datetime
from tqdm import tqdm
from config import beijing_lat_range, beijing_lon_range

print("脚本开始执行：Geolife轨迹数据预处理")

path = 'D:\dataset\geolife\Data\\'
dir_traj = os.listdir(path)

geolife_traj = [] # 用于存储从原始文件中读取的轨迹数据
print(f"开始遍历 {len(dir_traj)} 个用户目录以提取轨迹文件...")
for i, dirt in enumerate(tqdm(dir_traj)): 
    print(f"[{i+1}/{len(dir_traj)}] 正在处理用户目录: {dirt}")
    new_path = path + dirt + '\Trajectory\\' 
    files = os.listdir(new_path) 
    for j, file in enumerate(files):
        f = open(new_path + file) # 打开轨迹文件
        iter_f = iter(f) 
        tmp = [] # 临时列表，用于存储当前轨迹文件的所有行
        # 读取文件所有行
        for line in iter_f:
            tmp.append(line)
        # 删除文件头部的前6行元数据，只保留轨迹点数据
        del tmp[0]
        del tmp[0]
        del tmp[0]
        del tmp[0]
        del tmp[0]
        del tmp[0]
        geolife_traj.append(tmp) 
        f.close() 


Trajectory = [] # 存储最终处理后的轨迹数据
count = 0 # 轨迹计数器
print(f"开始解析和过滤 {len(geolife_traj)} 条原始轨迹...")
for i, trajs in enumerate(tqdm(geolife_traj)): 
    inrange = True # 标记当前轨迹是否在北京范围内
    Traj = [] 
    for traj in trajs:
        tr = traj.split(',') # 按逗号分割轨迹点数据
        lat = np.float64(tr[0]) # 提取纬度并转换为浮点数
        lon = np.float64(tr[1]) # 提取经度并转换为浮点数
        time1 = tr[-2] # 提取日期字符串
        time2 = tr[-1] # 提取时间字符串
        time1_list = time1.split('-') # 分割日期
        time2_list = time2.split(':')

        t = datetime.datetime(int(time1_list[0]), int(time1_list[1]), int(time1_list[2]),
                              int(time2_list[0]), int(time2_list[1]), int(time2_list[2]))

        t1 = ((t-datetime.datetime(1970, 1, 1)).total_seconds())

        # 检查轨迹点是否在北京经纬度范围之外
        if ((lat < beijing_lat_range[0]) | (lat > beijing_lat_range[1]) | (lon < beijing_lon_range[0]) | (lon > beijing_lon_range[1])):
            inrange = False # 如果有任何一个点超出范围，则标记为不在范围内
        traj_tup = (lon, lat, t1, t) # 创建轨迹点元组 (经度, 纬度, Unix时间戳, datetime对象)
        Traj.append(traj_tup)
        
    if inrange != False:
        Trajectory.append(Traj)
    else:
        print(f"  轨迹 {i+1} 被过滤，因为它超出了北京范围。")
    count += 1
        # 以下是被注释掉的代码，用于将轨迹数据写入文件，例如训练集和测试集
        # f = open('geolife_trajs', 'a')
        # f.write(str(count) + '\t' + str(len(Traj)) + '\t')
        # f.write(", ".join(str(x) for x in Traj))
        # f.close()
        # if(count < 1800):
        # \tf=open('geolife_train', 'a')
        # \tf.write(str(count) + '\t' + str(len(Traj)) + '\t')
        # \tf.write(", ".join(str(x) for x in Traj))
        # \tf.close()
        # if((count >= 1800) and (count <= 9000):
        # \tf=open('geolife_test', 'a')
        # \tf.write(str(count) + '\t' + str(len(Traj)) + '\t')
        # \tf.write(", ".join(str(x) for x in Traj))
        # \tf.close()

print(f"所有轨迹处理完毕。最终筛选出的轨迹数量：{len(Trajectory)}") # 16830

output_file = 'data/geolife/geolife'
print(f"正在将处理后的轨迹数据保存到文件: {output_file}")
with open(output_file, 'wb') as f:
    pickle.dump(Trajectory, f) # 使用pickle保存Trajectory对象
print("数据保存完成。脚本执行结束。")
