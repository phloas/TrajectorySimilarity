# 轨迹相似度任务研究

## 项目概述
本项目聚焦于轨迹相似度的系统性研究，涵盖数据预处理、三元组生成、轨迹距离计算、图结构构建、特征提取、数据集划分及模型训练等核心环节，旨在为轨迹数据的相似性度量与建模提供高效、可复用的工具链。

## 技术选型
- **主要编程语言**：Python 3.9+
- **数据处理**：Pickle（数据存储）、NumPy、Pandas
- **距离计算**：DTW、Hausdorff 距离、欧氏距离
- **图处理**：NetworkX、PyTorch Geometric
- **机器学习框架**：PyTorch
- **版本控制**：Git

## 项目结构 / 模块划分
- `/dataset`：存放 Geolife 与 Porto 两个真实轨迹数据集
- `/data`：存放处理好的 Geolife 与 Porto 两个轨迹数据集
- `/features`：存储轨迹特征表示
- `/tools`：通用工具函数
  - `distance_computation.py`：轨迹距离计算
  - `preprocess.py`：轨迹筛选与网格特征生成
- `/preprocess`：数据集预处理
  - `preprocess_geolife.py`：Geolife 数据集预处理，合并为 pickle 文件
  - `preprocess_porto.py`：Porto 数据集预处理，合并为 pickle 文件
- `dataset.py`：数据集划分、标签生成、距离矩阵处理
- `triplet_sampling.py`：训练集轨迹的三元组采样
- `preprocess.py`：轨迹对距离计算
- `generate_graph.py`：轨迹图结构生成
- `traj_model.py`：轨迹相似度模型实现
- `traj_train.py`：构建训练，验证，测试的Dataloader

## 数据模型
- **轨迹点 (Trajectory Point)**
  - 经度 (longitude)
  - 纬度 (latitude)
  - 时间戳 (timestamp)
- **轨迹 (Trajectory)**
  - 轨迹点序列
  - 轨迹ID
  - 轨迹特征
- **距离矩阵 (Distance Matrix)**
  - 空间距离矩阵
  - 时间距离矩阵

## 开发状态跟踪
| 模块/功能                   | 状态     | 负责人 | 计划完成日期 | 实际完成日期 | 关联PR/Commit | 备注与问题追踪                                   |
|----------------------------|----------|--------|--------------|--------------|---------------|-----------------------------------------------|
| Geolife 数据预处理         | 已完成   | AI/您  | YYYY-MM-DD   | 2024-06-11   |               | 已实现命令行参数解析和轨迹筛选                 |
| Porto 数据预处理           | 已完成   | AI/您  | YYYY-MM-DD   | 2024-06-11   |               | 已实现命令行参数解析和轨迹筛选                 |
| 轨迹筛选与特征生成         | 已完成   | AI/您  | YYYY-MM-DD   | 2024-06-11   |               | 已实现轨迹筛选和特征提取                       |
| 轨迹距离计算               | 已完成   | AI/您  | YYYY-MM-DD   | 2024-06-11   |               | 已实现多种距离度量方法                         |
| 训练集三元组采样逻辑独立   | 已完成   | AI/您  | YYYY-MM-DD   | 2024-06-12   |               | triplet_sampling.py 独立采样逻辑，dataset.py 调用 |
| 图结构生成                 | 已完成   | AI/您  | YYYY-MM-DD   | 2024-06-11   |               | 已实现图结构生成和特征提取                     |
| 数据集处理                 | 已完成   | AI/您  | YYYY-MM-DD   | 2024-06-11   |               | 已实现数据集划分和标签生成                     |
| 模型训练                   | 进行中   | AI/您  | YYYY-MM-DD   |              |               | 正在实现轨迹相似度学习模型                     |

## 代码检查与问题记录
1. `dataset.py` 代码优化，移除冗余日志前缀
2. 优化数据加载与保存的异常处理
3. 距离矩阵归一化逻辑改进

## 部署指南

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
   python preprocess/preprocess_geolife.py
   python preprocess/preprocess_porto.py
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
   python train.py
   ```

## 可更新的内容
### 模型内容的更新
1. 构建Cross-encoder和Bi-encoder
2. Local Mask

### 多种三元组采样的方法