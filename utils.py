import torch
import numpy as np
import random
import matplotlib.pyplot as plt

def set_seed(seed=42):
    """锁定随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f">>> 随机种子已锁定: {seed}")

def enable_dropout(m):
    """仅开启 Dropout 层，保持 BN/LN 层为 eval 模式"""
    for module in m.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()

def truncate_curve_fine(y_data, max_strain=0.5):
    """精细截断函数：从峰值后寻找第一次归零点"""
    if isinstance(y_data, torch.Tensor):
        y_data = y_data.cpu().numpy().flatten()
    
    x_axis = np.linspace(0, max_strain, len(y_data))
    peak_idx = np.argmax(y_data)
    peak_val = y_data[peak_idx]
    
    # 极低阈值
    zero_threshold = max(peak_val * 0.005, 0.005)
    cut_idx = len(y_data)
    
    start_search = max(peak_idx + 1, 0)
    for i in range(start_search, len(y_data)):
        if y_data[i] < zero_threshold:
            if i + 1 < len(y_data) and y_data[i+1] > zero_threshold:
                continue 
            cut_idx = i
            break
            
    return x_axis[:cut_idx], y_data[:cut_idx]

def truncate_curve_smart(y_data, max_strain=0.5):
    """智能截断函数：逆向搜索"""
    if isinstance(y_data, torch.Tensor):
        y_data = y_data.cpu().numpy().flatten()
    
    n_points = len(y_data)
    x_axis = np.linspace(0, max_strain, n_points)
    
    peak_idx = np.argmax(y_data)
    peak_val = y_data[peak_idx]
    noise_threshold = max(peak_val * 0.02, 0.02)
    
    cut_idx = n_points 
    for i in range(n_points - 1, peak_idx, -1):
        if y_data[i] > noise_threshold:
            cut_idx = i + 1 
            break
            
    cut_idx = max(cut_idx, peak_idx + 2)
    cut_idx = min(cut_idx, n_points) 
    
    return x_axis[:cut_idx], y_data[:cut_idx]