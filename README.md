# 轨迹相似度任务研究

## 项目概括
本项目旨在研究轨迹相似度任务。它涵盖了轨迹数据集的预处理、三元组的生成、轨迹之间距离的计算、轨迹图的生成、图特征提取、数据集划分以及模型训练等多个核心模块。

## 技术选型
- 主要编程语言: Python
- 数据处理: 
  - Pickle (用于合并和存储轨迹数据)
  - NumPy (用于数值计算和数组操作)
  - Pandas (用于数据处理和分析)
- 距离计算: 
  - DTW (Dynamic Time Warping)
  - Hausdorff 距离
  - 欧氏距离
- 图处理: 
  - NetworkX (用于图结构处理)
  - PyTorch Geometric (用于图神经网络)
- 机器学习框架: PyTorch
- 版本控制: Git

## 项目结构 / 模块划分
- `/data`: 保存两个现实生活的轨迹数据集 geolife 和 porto
- `/ground_truth`: 存储轨迹对之间的真实距离矩阵
- `/features`: 存储轨迹的特征表示
- `/tools`: 保存工具函数
  - `distance_computation.py`: 用于计算两个轨迹之间的距离
  - `preprocess.py`: 用于对数据集中的轨迹进行筛选，并生成轨迹的网格特征
- `preprocess_geolife.py`: 对 geolife 数据集进行预处理，将原始轨迹数据合并成一个 pickle 文件
- `preprocess_porto.py`: 对 porto 数据集进行预处理，将原始轨迹数据合并成一个 pickle 文件
- `preprocess.py`: 用于预处理潜在的三元组，并生成轨迹对之间的距离
- `generate_graph.py`: 用于生成轨迹图结构
- `dataset.py`: 用于数据集的处理、划分和标签生成
- `traj_model.py`: 轨迹相似度模型的实现
- `pars_args.py`: 命令行参数解析

## 核心功能 / 模块详解
- **数据预处理:**
  - `preprocess_geolife.py`: 处理 Geolife 轨迹数据集，生成统一的 pickle 文件
  - `preprocess_porto.py`: 处理 Porto 轨迹数据集，生成统一的 pickle 文件
- **轨迹筛选与特征生成:**
  - `tools/preprocess.py`: 负责轨迹的筛选和网格特征的生成
- **距离计算:**
  - `tools/distance_computation.py`: 提供轨迹之间距离计算的功能，支持多种距离度量方法
- **三元组与距离对生成:**
  - `preprocess.py`: 负责生成潜在的三元组以及轨迹对之间的距离
- **图结构生成:**
  - `generate_graph.py`: 基于轨迹数据生成图结构，用于后续的图特征提取
- **数据集处理:**
  - `dataset.py`: 负责数据集的划分、标签生成和距离矩阵处理
- **模型训练:**
  - `traj_model.py`: 实现轨迹相似度学习模型

## 数据模型
- **轨迹点 (Trajectory Point):**
  - 经度 (longitude)
  - 纬度 (latitude)
  - 时间戳 (timestamp)
- **轨迹 (Trajectory):**
  - 轨迹点序列
  - 轨迹ID
  - 轨迹特征
- **距离矩阵 (Distance Matrix):**
  - 空间距离矩阵
  - 时间距离矩阵

## 开发状态跟踪
| 模块/功能 | 状态 | 负责人 | 计划完成日期 | 实际完成日期 | 关联PR/Commit | 备注与问题追踪 |
|---|---|---|---|---|---|---|
| Geolife 数据预处理 | 已完成 | AI/您 | YYYY-MM-DD | 2024-06-11 | | 已实现命令行参数解析和轨迹筛选 |
| Porto 数据预处理 | 已完成 | AI/您 | YYYY-MM-DD | 2024-06-11 | | 已实现命令行参数解析和轨迹筛选 |
| 轨迹筛选与网格特征生成 | 已完成 | AI/您 | YYYY-MM-DD | 2024-06-11 | | 已实现轨迹筛选和特征提取 |
| 轨迹距离计算 | 已完成 | AI/您 | YYYY-MM-DD | 2024-06-11 | | 已实现多种距离度量方法 |
| 三元组与轨迹对距离生成 | 已完成 | AI/您 | YYYY-MM-DD | 2024-06-11 | | 已实现三元组生成和距离计算 |
| 图结构生成 | 已完成 | AI/您 | YYYY-MM-DD | 2024-06-11 | | 已实现图结构生成和特征提取 |
| 数据集处理 | 已完成 | AI/您 | YYYY-MM-DD | 2024-06-11 | | 已实现数据集划分和标签生成 |
| 模型训练 | 进行中 | AI/您 | YYYY-MM-DD | | | 正在实现轨迹相似度学习模型 |

## 代码检查与问题记录
1. 已完成对 `dataset.py` 的代码优化，移除了冗余的日志前缀
2. 优化了数据加载和保存的错误处理机制
3. 改进了距离矩阵的归一化处理逻辑

## 环境设置与部署指南
### 环境要求
- Python 3.8+
- PyTorch 1.8+
- NumPy
- Pandas
- NetworkX
- PyTorch Geometric

### 安装步骤
1. 克隆项目仓库
2. 安装依赖包：
   ```bash
   pip install -r requirements.txt
   ```
3. 准备数据集：
   - 将 Geolife 数据集放入 `/data/geolife` 目录
   - 将 Porto 数据集放入 `/data/porto` 目录

### 使用说明
1. 数据预处理：
   ```bash
   python preprocess_geolife.py
   python preprocess_porto.py
   ```
2. 生成轨迹图：
   ```bash
   python generate_graph.py
   ```
3. 处理数据集：
   ```bash
   python dataset.py
   ```
4. 训练模型：
   ```bash
   python traj_model.py
   ```