import os
import torch

# ================= 路径配置 =================
# 请根据实际情况修改 BASE_PATH
BASE_PATH = r'D:\ABAQUS2024\temp\Virtual_Growth\Spiderweb\R2\Datas'
FILE_FORCE = os.path.join(BASE_PATH, 'Force.xlsx')
FILE_SUMMARY = os.path.join(BASE_PATH, 'Summary.xlsx')
FILE_COMBO = os.path.join(BASE_PATH, 'Combinations.xlsx')

# 模型与Scaler保存路径
MODEL_SAVE_PATH = 'spiderweb_model.pth'
SCALER_X_PATH = 'scaler_X.pkl'
SCALER_Y_SCALAR_PATH = 'scaler_Y_scalars.pkl'
SCALER_Y_CURVE_PATH = 'scaler_Y_curve.pkl'
DATA_STATS_PATH = 'data_stats.pkl'  # [新增] 保存最佳截断点统计

# ================= 物理常数 =================
LENGTH_MM = 30.0
AREA_MM2 = 30.0
NUM_POINTS = 200
# MAX_STRAIN = 0.5 # [注释掉] 实际上将由 dataset.py 动态计算覆盖

# ================= 训练超参数 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1000 
LR = 0.01
EPOCHS = 2500
PATIENCE = 200

# ================= 优化/反求参数 =================
LAYER_MIN, LAYER_MAX = 5, 20
SPACING_MIN, SPACING_MAX = 0.5, 5.0
WIDTH_MIN, WIDTH_MAX = 0.1, 2.0
N_RADIALS_MIN, N_RADIALS_MAX = 4, 36