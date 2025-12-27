import pandas as pd
import numpy as np
import os
import torch
import joblib
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import config

class AugmentedDataset(Dataset):
    def __init__(self, X, y_curve, y_scalar, mask, noise_level=0.05, augment=False):
        self.X = torch.FloatTensor(X)
        self.y_curve = torch.FloatTensor(y_curve)
        self.y_scalar = torch.FloatTensor(y_scalar)
        self.mask = torch.FloatTensor(mask)
        self.noise_level = noise_level
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            noise = torch.randn_like(x) * self.noise_level
            x = x + noise
        return x, self.y_curve[idx], self.y_scalar[idx], self.mask[idx]

def analyze_fracture_limits(df_combo, df_force, summary_map, disp_raw):
    """
    [动态截断逻辑] 
    扫描全数据集，找到覆盖 99% 样本的有效应变范围，减少 0 值填充。
    """
    print(">>> 正在进行全数据集扫描，寻找最佳应变截断点 (Optimal Cutoff)...")
    fracture_strains = []
    
    for idx, row in df_combo.iterrows():
        job_name = f"Job-{int(row.iloc[0])}"
        if job_name not in df_force.columns: continue
        
        force_vals = df_force[job_name].values
        mask = ~np.isnan(force_vals) & ~np.isnan(disp_raw)
        d_clean = disp_raw[mask]
        f_clean = force_vals[mask]
        
        if len(d_clean) < 10: continue
        
        strain = d_clean / config.LENGTH_MM
        stress = f_clean / config.AREA_MM2
        
        # 简单判定断裂：寻找峰值后归零点
        if len(stress) > 0:
            peak_idx = np.argmax(stress)
            post_peak = stress[peak_idx:]
            zeros = np.where(post_peak <= 1e-3)[0]
            if len(zeros) > 0:
                cutoff_strain = strain[peak_idx + zeros[0]]
                fracture_strains.append(cutoff_strain)
            else:
                fracture_strains.append(strain[-1])
                
    if not fracture_strains:
        return 0.5
        
    p99 = np.percentile(fracture_strains, 99)
    optimal_limit = min(p99 * 1.05, 0.6) # 留 5% 余量，且不超过 0.6
    
    print(f">>> 统计完成: P99断裂应变={p99:.4f} -> 设定 Max Strain={optimal_limit:.4f}")
    return optimal_limit

def load_raw_data():
    if not os.path.exists(config.FILE_FORCE):
        print(f"!!! 错误: 未找到 Force 文件")
        # Mock fallback
        N = 100
        return np.random.rand(N, 13), np.random.rand(N, 200), np.random.rand(N, 4), np.ones((N, 200)), [f"Job-{i}" for i in range(N)], 0.5

    try:
        df_force = pd.read_excel(config.FILE_FORCE)
        df_combo = pd.read_excel(config.FILE_COMBO)
        df_summary = pd.read_excel(config.FILE_SUMMARY)
    except Exception as e:
        print(f"读取 Excel 出错: {e}")
        return None, None, None, None, None, None
    
    summary_map = {}
    for i, row in df_summary.iterrows():
        job_name = row.iloc[0]
        targets = row.iloc[1:5].values.astype(float)
        summary_map[job_name] = targets
        
    disp_raw = df_force.iloc[:, 0].values
    
    # 1. 获取动态截断点
    dynamic_max_strain = analyze_fracture_limits(df_combo, df_force, summary_map, disp_raw)
    
    X_list, y_curve_list, y_scalars_list, mask_list, names_list = [], [], [], [], []
    processed_count = 0
    
    for idx, row in df_combo.iterrows():
        job_id = int(row.iloc[0])
        job_name = f"Job-{job_id}"
        
        if job_name in df_force.columns and job_name in summary_map:
            features = row.iloc[1:14].values.astype(float)
            scalars = summary_map[job_name]
            
            force_vals = df_force[job_name].values
            mask = ~np.isnan(force_vals) & ~np.isnan(disp_raw)
            d_clean = disp_raw[mask]
            f_clean = force_vals[mask]
            
            if len(d_clean) < 10: continue
            
            strain = d_clean / config.LENGTH_MM
            stress = f_clean / config.AREA_MM2
            
            if len(stress) > 0:
                peak_idx = np.argmax(stress)
                post_peak_zeros = np.where(stress[peak_idx:] <= 0)[0]
                if len(post_peak_zeros) > 0:
                    cutoff_idx = peak_idx + post_peak_zeros[0]
                    strain = strain[:cutoff_idx+1]
                    stress = stress[:cutoff_idx+1]
                    stress[-1] = 0.0 
            stress[stress < 0] = 0
            
            # 2. 使用动态截断点进行插值
            f_interp = interp1d(strain, stress, kind='linear', bounds_error=False, fill_value=0.0)
            y_new = f_interp(np.linspace(0, dynamic_max_strain, config.NUM_POINTS))
            
            curve_mask = (y_new > 1e-6).astype(float)
            curve_mask[:5] = 1.0 
            
            X_list.append(features)
            y_curve_list.append(y_new)
            y_scalars_list.append(scalars)
            mask_list.append(curve_mask)
            names_list.append(job_name)
            processed_count += 1
            
    return np.array(X_list), np.array(y_curve_list), np.array(y_scalars_list), np.array(mask_list), np.array(names_list), dynamic_max_strain

def get_data_bundle(is_training=True):
    res = load_raw_data()
    if res[0] is None: return None
    X, y_curve, y_scalars, masks, names, dynamic_max_strain = res

    X_temp, X_test, yc_temp, yc_test, ys_temp, ys_test, m_temp, m_test, n_temp, n_test = train_test_split(
        X, y_curve, y_scalars, masks, names, test_size=0.1, random_state=42
    )
    X_train, X_val, yc_train, yc_val, ys_train, ys_val, m_train, m_val, n_train, n_val = train_test_split(
        X_temp, yc_temp, ys_temp, m_temp, n_temp, test_size=1/9, random_state=42
    )

    if is_training:
        print(">>> Fitting Scalers...")
        scaler_X = MinMaxScaler().fit(X_train)
        scaler_Y_curve = MinMaxScaler().fit(yc_train)
        scaler_Y_scalar = MinMaxScaler().fit(ys_train)
        
        joblib.dump(scaler_X, config.SCALER_X_PATH)
        joblib.dump(scaler_Y_scalar, config.SCALER_Y_SCALAR_PATH)
        joblib.dump(scaler_Y_curve, config.SCALER_Y_CURVE_PATH)
        joblib.dump(dynamic_max_strain, config.DATA_STATS_PATH) # 保存动态截断点
    else:
        print(">>> Loading Scalers...")
        try:
            scaler_X = joblib.load(config.SCALER_X_PATH)
            scaler_Y_curve = joblib.load(config.SCALER_Y_CURVE_PATH)
            scaler_Y_scalar = joblib.load(config.SCALER_Y_SCALAR_PATH)
            if os.path.exists(config.DATA_STATS_PATH):
                dynamic_max_strain = joblib.load(config.DATA_STATS_PATH)
                print(f">>> 加载历史 Max Strain: {dynamic_max_strain:.4f}")
            else:
                dynamic_max_strain = 0.5 
        except FileNotFoundError:
            return None

    return {
        'X_train': scaler_X.transform(X_train), 'y_curve_train': scaler_Y_curve.transform(yc_train), 'y_scalar_train': scaler_Y_scalar.transform(ys_train), 'mask_train': m_train,
        'X_val': scaler_X.transform(X_val),     'y_curve_val': scaler_Y_curve.transform(yc_val),     'y_scalar_val': scaler_Y_scalar.transform(ys_val), 'mask_val': m_val,
        'X_test': scaler_X.transform(X_test),   'y_curve_test': scaler_Y_curve.transform(yc_test),   'y_scalar_test': scaler_Y_scalar.transform(ys_test), 'mask_test': m_test,
        'X_test_raw': X_test, 'y_curve_test_raw': yc_test, 'y_scalar_test_raw': ys_test,
        'names_test': n_test,
        'scalers': (scaler_X, scaler_Y_curve, scaler_Y_scalar),
        'max_strain': dynamic_max_strain
    }