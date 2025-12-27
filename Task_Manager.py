import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import random
import os
import joblib

import config
import Model_Definition as model_lib
import dataset
import utils

# ================= [核心修改] 截断与物理量计算 =================

def calc_metrics_truncated(stress, strain):
    """
    计算物理属性的稳健版本：
    1. 自动截断：识别断裂点，去除尾部噪音 (Ghost Tail)。
    2. 积分：使用 np.trapz 计算 MOT。
    3. [修改] 位移：计算峰值应力对应的位移 (Peak Displacement)。
    """
    # 1. 寻找峰值
    peak_idx = np.argmax(stress)
    peak_val = stress[peak_idx]
    
    # 2. 智能截断逻辑 (用于 MOT 积分截止)
    # 从峰值向后找，一旦应力掉到峰值的 10% 以下或 0.05 MPa 以下，视为断裂
    cutoff_idx = len(stress) - 1
    threshold = max(peak_val * 0.10, 0.05)
    
    for i in range(peak_idx + 1, len(stress)):
        if stress[i] < threshold:
            cutoff_idx = i
            break
            
    # 3. 截取有效片段 (用于 MOT 计算)
    valid_stress = stress[:cutoff_idx+1]
    valid_strain = strain[:cutoff_idx+1]
    
    # 4. 计算 MOT (KJ/m^3)
    if len(valid_stress) > 1:
        mot = np.trapz(valid_stress, valid_strain) * 1000.0
    else:
        mot = 0.0
        
    # 5. [修改点] 计算位移 (Displacement at Peak Load)
    # 逻辑：找到峰值索引，取该索引对应的应变，乘以长度
    if peak_idx < len(strain):
        disp = strain[peak_idx] * config.LENGTH_MM 
    else:
        disp = 0.0
    
    return mot, peak_val, disp

# ================= 1. 训练模块 =================

def train_model():
    """对应代码一：训练主流程 (适配 Normalized Masked R2 + Robust Metrics)"""
    bundle = dataset.get_data_bundle(is_training=True)
    if not bundle: return

    # [新增] 获取动态计算的最大应变
    current_max_strain = bundle['max_strain']
    print(f"\n>>> [Info] 本次训练物理空间范围 (Strain): 0 ~ {current_max_strain:.4f}")

    device = config.DEVICE
    
    train_ds = dataset.AugmentedDataset(bundle['X_train'], bundle['y_curve_train'], bundle['y_scalar_train'], bundle['mask_train'], augment=False)
    val_ds = dataset.AugmentedDataset(bundle['X_val'], bundle['y_curve_val'], bundle['y_scalar_val'], bundle['mask_val'], augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE)
    
    net = model_lib.SpiderWebModel().to(device)
    optimizer = optim.AdamW(net.parameters(), lr=config.LR, weight_decay=1e-3) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)
    mse_loss_fn = nn.MSELoss() 
    
    best_loss = float('inf')
    counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    print(f">>> 开始训练 (MinMax Scaler + Masked Loss)...")
    
    for epoch in range(config.EPOCHS):
        net.train()
        loss_epoch = 0.0
        for bx, byc, bys, bmask in train_loader:
            bx, byc, bys, bmask = bx.to(device), byc.to(device), bys.to(device), bmask.to(device)
            optimizer.zero_grad()
            pc, ps = net(bx)
            diff = (pc - byc) ** 2 
            loss_c = (diff * bmask).sum() / (bmask.sum() + 1e-8)
            loss_s = mse_loss_fn(ps, bys)
            loss = loss_c + 1.0 * loss_s
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
        
        avg_train_loss = loss_epoch / len(train_loader)
        
        # Validation
        net.eval()
        loss_val = 0.0
        with torch.no_grad():
            for bx, byc, bys, bmask in val_loader:
                bx, byc, bys, bmask = bx.to(device), byc.to(device), bys.to(device), bmask.to(device)
                pc, ps = net(bx)
                diff = (pc - byc) ** 2
                loss_c = (diff * bmask).sum() / (bmask.sum() + 1e-8)
                loss_s = mse_loss_fn(ps, bys)
                loss = loss_c + 1.0 * loss_s
                loss_val += loss.item()
        
        avg_val_loss = loss_val / len(val_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            counter = 0
            torch.save(net.state_dict(), config.MODEL_SAVE_PATH)
        else:
            counter += 1
            if counter >= config.PATIENCE:
                print(f"Early Stopping at Epoch {epoch}")
                break
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:04d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
            
    print(">>> 训练结束，模型已保存。")
    
    # ------------------ 最终测试 (应用 Mask & Normalization 优势) ------------------
    print("\n>>> 正在进行最终测试集评估...")
    net.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    net.eval()
    
    test_ds = dataset.AugmentedDataset(bundle['X_test'], bundle['y_curve_test'], bundle['y_scalar_test'], bundle['mask_test'], augment=False)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE)
    
    ys_true_list, ps_raw_list = [], [] 
    yc_true_list, yc_pred_list = [], [] 
    mask_true_list = []
    
    test_loss_total = 0.0
    with torch.no_grad():
        for bx, byc, bys, bmask in test_loader:
            bx, byc, bys, bmask = bx.to(device), byc.to(device), bys.to(device), bmask.to(device)
            pc, ps = net(bx)
            
            diff = (pc - byc) ** 2
            loss_c = (diff * bmask).sum() / (bmask.sum() + 1e-8)
            loss_s = mse_loss_fn(ps, bys)
            test_loss_total += (loss_c + loss_s).item()
            
            ys_true_list.append(bys.cpu().numpy())
            ps_raw_list.append(ps.cpu().numpy())
            yc_true_list.append(byc.cpu().numpy())
            yc_pred_list.append(pc.cpu().numpy())
            mask_true_list.append(bmask.cpu().numpy())
            
    avg_test_loss = test_loss_total / len(test_loader)
    
    # 1. 准备归一化数据 (用于计算 Curve R2)
    yc_true_norm = np.vstack(yc_true_list)
    yc_pred_norm = np.vstack(yc_pred_list)
    mask_true_np = np.vstack(mask_true_list)
    
    # 2. 准备真实物理数据 (用于计算 Scalar R2)
    _, scaler_Y_curve, scaler_Y_scalar = bundle['scalers']
    ys_true_norm = np.vstack(ys_true_list)
    ys_true_real = scaler_Y_scalar.inverse_transform(ys_true_norm)
    
    # 3. [关键步骤] 计算 Curve R2 (归一化 + Masked)
    yc_true_flat = yc_true_norm.flatten()
    yc_pred_flat = yc_pred_norm.flatten()
    mask_flat = mask_true_np.flatten()
    
    valid_indices = mask_flat > 0.5
    if valid_indices.sum() > 0:
        r2_c_final = r2_score(yc_true_flat[valid_indices], yc_pred_flat[valid_indices])
    else:
        r2_c_final = 0.0
        
    # 4. [关键步骤] 计算物理属性 R2 (使用动态截断 + Peak Displacement)
    yc_pred_real = scaler_Y_curve.inverse_transform(yc_pred_norm)
    
    # [修改点] 使用动态计算的 max_strain 生成 x 轴
    x_axis = np.linspace(0, current_max_strain, config.NUM_POINTS)
    
    derived_mot, derived_peak, derived_disp = [], [], []
    for i in range(len(yc_pred_real)):
        m, p, d = calc_metrics_truncated(yc_pred_real[i], x_axis)
        derived_mot.append(m)
        derived_peak.append(p)
        derived_disp.append(d)
        
    # Volume 仍使用标量头预测
    ps_norm = np.vstack(ps_raw_list)
    ps_real = scaler_Y_scalar.inverse_transform(ps_norm)
    pred_vol = ps_real[:, 3]
    
    r2_mot = r2_score(ys_true_real[:, 0], derived_mot)
    r2_peak = r2_score(ys_true_real[:, 1], derived_peak)
    r2_disp = r2_score(ys_true_real[:, 2], derived_disp)
    r2_vol = r2_score(ys_true_real[:, 3], pred_vol)
    
    print(f"\n[最终性能报告 - Robust R2]")
    print(f"-"*40)
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"R2 (MOT):  {r2_mot:.4f} (Derived from Curve)")
    print(f"R2 (Peak): {r2_peak:.4f} (Derived from Curve)")
    print(f"R2 (Disp): {r2_disp:.4f} (Peak Displacement)")
    print(f"R2 (Vol):  {r2_vol:.4f} (Direct Prediction)")
    print(f"Curve R2:  {r2_c_final:.4f} (Normalized & Masked)")
    print(f"="*40)
    
    plt.figure(figsize=(12, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.yscale('log')
    plt.legend()
    plt.show()

# ================= 2. 预测模块 =================

def run_prediction():
    """对应代码二：正向预测 (适配 Normalized R2 + Truncated Metrics)"""
    bundle = dataset.get_data_bundle(is_training=False)
    if not bundle: return
    scaler_X, scaler_Y_curve, scaler_Y_scalar = bundle['scalers']
    X_test = bundle['X_test']
    names_test = bundle['names_test']
    mask_test = bundle['mask_test']
    
    # [修改点] 获取动态 max_strain
    current_max_strain = bundle['max_strain']

    device = config.DEVICE
    net = model_lib.SpiderWebModel().to(device)
    try:
        net.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    except:
        print("Model not found!")
        return
    net.eval()
    
    sample_idx = random.randint(0, len(X_test) - 1)
    real_job_name = names_test[sample_idx]
    mask_s = mask_test[sample_idx] # (Points,)
    
    print(f"\n" + "="*50)
    print(f">>> 正在验证测试集样本索引: {sample_idx}")
    print(f">>> 对应原始 Excel ID: 【 {real_job_name} 】")
    print(f"="*50)
    
    input_params = X_test[sample_idx]
    gt_curve_norm = bundle['y_curve_test'][sample_idx] # Normalized Ground Truth
    
    inp_tensor = torch.from_numpy(np.array([input_params])).float().to(device)
    
    # Forward
    with torch.no_grad():
        pc_norm_s, ps_norm_s = net(inp_tensor)
        pred_curve_norm = pc_norm_s.cpu().numpy()[0]
        
    # --- [关键改进] 计算 Curve R2 (Normalized + Masked) ---
    valid_mask = mask_s > 0.5
    if valid_mask.sum() > 0:
        # 只取 Mask 有效的部分进行 R2 计算
        r2_curve_sample = r2_score(gt_curve_norm[valid_mask], pred_curve_norm[valid_mask])
    else:
        r2_curve_sample = 0.0

    # 反归一化 (用于绘图和物理计算)
    pred_curve_real = scaler_Y_curve.inverse_transform(pred_curve_norm.reshape(1, -1))[0]
    gt_curve_real = scaler_Y_curve.inverse_transform(gt_curve_norm.reshape(1, -1))[0]
    
    # [修改点] 计算物理属性 (使用动态截断 + Peak Disp)
    x_axis = np.linspace(0, current_max_strain, config.NUM_POINTS)
    p_mot, p_peak, p_disp = calc_metrics_truncated(pred_curve_real, x_axis)
    
    # 获取 Vol (Index 3)
    ps_real = scaler_Y_scalar.inverse_transform(ps_norm_s.cpu().numpy())[0]
    p_vol = ps_real[3]
    
    # 获取 Ground Truth Scalar
    gt_scalar_s = bundle['y_scalar_test'][sample_idx]
    gt_scalar = scaler_Y_scalar.inverse_transform(gt_scalar_s.reshape(1, -1))[0]

    # 绘图 (Masked)
    x_plot = x_axis[valid_mask]
    y_pred_plot = pred_curve_real[valid_mask]
    y_gt_plot = gt_curve_real[valid_mask]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_pred_plot, 'b-', label='AI Prediction')
    plt.plot(x_plot, y_gt_plot, 'r--', label='Ground Truth')
    plt.title(f"Prediction: {real_job_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 打印对比
    print(f"{'='*40}")
    print(f">>> 物理属性预测对比 (Derived from Curve):")
    print(f"Curve R2 (Norm & Masked): {r2_curve_sample:.4f}")
    print(f"{'-'*46}")
    print(f"{'Metric':<10} | {'Prediction':<10} | {'Ground Truth':<10} | {'Error':<10}")
    print(f"{'-'*46}")
    
    metrics = [('MOT', p_mot), ('Peak', p_peak), ('Disp', p_disp), ('Vol', p_vol)]
    
    for i, (name, val) in enumerate(metrics):
        t_val = gt_scalar[i]
        error = abs(val - t_val) / (abs(t_val) + 1e-6) * 100 
        print(f"{name:<10} | {val:<10.4f} | {t_val:<10.4f} | {error:<9.2f}%")
    print(f"{'='*40}")
    
    plt.show()

# ================= 3. 优化模块 =================

def run_optimization():
    """对应代码三：多目标优化 (适配 Robust Metrics Reporting)"""
    utils.set_seed(42)
    bundle = dataset.get_data_bundle(is_training=False)
    if not bundle: return
    
    scaler_X, scaler_Y_curve, scaler_Y_scalar = bundle['scalers']
    
    # [修改点] 获取动态 max_strain
    current_max_strain = bundle['max_strain']
    print(f">>> 优化模块加载的 Max Strain: {current_max_strain:.4f}")

    device = config.DEVICE
    net = model_lib.SpiderWebModel().to(device)
    net.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    net.eval()
    
    # MinMax Params
    x_scale = torch.FloatTensor(scaler_X.scale_).to(device)
    x_min = torch.FloatTensor(scaler_X.min_).to(device)
    yc_scale = torch.FloatTensor(scaler_Y_curve.scale_).to(device)
    yc_min = torch.FloatTensor(scaler_Y_curve.min_).to(device)
    ys_scale = torch.FloatTensor(scaler_Y_scalar.scale_).to(device)
    ys_min = torch.FloatTensor(scaler_Y_scalar.min_).to(device)
    
    # [修改点] 积分轴基于 current_max_strain
    strain_axis = torch.linspace(0, current_max_strain, config.NUM_POINTS).to(device)
    
    # 优化设置
    BATCH_SIZE = 5000
    LR = 0.05
    STEPS = 1000
    
    init_tensor = torch.randn(BATCH_SIZE, 13).to(device)
    init_tensor.requires_grad = True
    optimizer = optim.Adam([init_tensor], lr=LR)
    
    valid_n = torch.tensor([4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36], device=device).float()
    perf_history = []
    
    def snap_to_valid_n(n_tensor):
        dist = torch.abs(n_tensor.unsqueeze(1) - valid_n.unsqueeze(0))
        min_indices = torch.argmin(dist, dim=1)
        return valid_n[min_indices]

    print(f">>> 开始优化 ({BATCH_SIZE} 组)...")

    for step in range(STEPS):
        optimizer.zero_grad()
        
        # 参数生成与归一化
        layers = torch.sigmoid(init_tensor[:, 0]) * (config.LAYER_MAX - config.LAYER_MIN) + config.LAYER_MIN
        spacing = torch.sigmoid(init_tensor[:, 1]) * (config.SPACING_MAX - config.SPACING_MIN) + config.SPACING_MIN
        n_rad = torch.sigmoid(init_tensor[:, 2]) * (config.N_RADIALS_MAX - config.N_RADIALS_MIN) + config.N_RADIALS_MIN
        angle = 360.0 / n_rad
        sw = torch.sigmoid(init_tensor[:, 3]) * (config.WIDTH_MAX - config.WIDTH_MIN) + config.WIDTH_MIN
        rw = torch.sigmoid(init_tensor[:, 4]) * (config.WIDTH_MAX - config.WIDTH_MIN) + config.WIDTH_MIN
        fin = torch.softmax(init_tensor[:, 5:9], dim=1)
        fout = torch.softmax(init_tensor[:, 9:13], dim=1)
        
        design = torch.cat([layers.unsqueeze(1), spacing.unsqueeze(1), angle.unsqueeze(1), 
                            sw.unsqueeze(1), rw.unsqueeze(1), fin, fout], dim=1)
        
        d_norm = design * x_scale + x_min
        pc_norm, ps_norm = net(d_norm)
        
        # 反归一化计算 Loss (Tensor operations)
        pc_real = pc_norm * yc_scale + yc_min
        ps_real = ps_norm * ys_scale + ys_min
        
        # Tensor积分 (简单梯形近似, 用于梯度回传)
        dx = strain_axis[1] - strain_axis[0]
        mot_real = torch.sum((pc_real[:, :-1] + pc_real[:, 1:]) / 2.0 * dx, dim=1) * 1000.0
        peak_real, _ = torch.max(pc_real, dim=1)
        vol_real = ps_real[:, 3]
        
        W_MOT, W_PEAK, W_VOL = 1.0, 2.0, 20.0
        perf_loss = -W_MOT * mot_real - W_PEAK * peak_real + W_VOL * vol_real
        
        int_penalty = 10.0 * (torch.sin(n_rad * torch.pi)**2 + torch.sin(layers * torch.pi)**2).mean()
        vol_limit = torch.relu(vol_real - 0.6) * 100.0
        
        loss = perf_loss.mean() + int_penalty + vol_limit.mean()
        loss.backward()
        optimizer.step()
        
        perf_history.append((W_MOT*mot_real + W_PEAK*peak_real - W_VOL*vol_real).mean().item())
        
        if (step+1) % 500 == 0:
            print(f"Step {step+1} | Loss: {loss.item():.4f}")

    # --- 结果提取与报告 ---
    with torch.no_grad():
        # 离散化参数
        n_final_cont = torch.sigmoid(init_tensor[:, 2]) * (config.N_RADIALS_MAX - config.N_RADIALS_MIN) + config.N_RADIALS_MIN
        n_valid_mapped = snap_to_valid_n(n_final_cont)
        angle_final = 360.0 / n_valid_mapped
        l_final = torch.round(torch.sigmoid(init_tensor[:, 0]) * (config.LAYER_MAX - config.LAYER_MIN) + config.LAYER_MIN)
        
        spacing_final = torch.sigmoid(init_tensor[:, 1]) * (config.SPACING_MAX - config.SPACING_MIN) + config.SPACING_MIN
        sw_final = torch.sigmoid(init_tensor[:, 3]) * (config.WIDTH_MAX - config.WIDTH_MIN) + config.WIDTH_MIN
        rw_final = torch.sigmoid(init_tensor[:, 4]) * (config.WIDTH_MAX - config.WIDTH_MIN) + config.WIDTH_MIN
        fin_final = torch.softmax(init_tensor[:, 5:9], dim=1)
        fout_final = torch.softmax(init_tensor[:, 9:13], dim=1)
        
        final_design_phys = torch.cat([
            l_final.unsqueeze(1), spacing_final.unsqueeze(1), angle_final.unsqueeze(1),
            sw_final.unsqueeze(1), rw_final.unsqueeze(1), fin_final, fout_final
        ], dim=1)
        
        final_design_norm = final_design_phys * x_scale + x_min
        final_curve_norm, final_ps_norm = net(final_design_norm)
        
        # 结果反归一化
        final_curve_real = scaler_Y_curve.inverse_transform(final_curve_norm.cpu().numpy())
        final_ps_real = scaler_Y_scalar.inverse_transform(final_ps_norm.cpu().numpy())
        final_vol = final_ps_real[:, 3]
        
        # [修改点] 使用 current_max_strain
        x_axis = np.linspace(0, current_max_strain, config.NUM_POINTS)
        final_scores = []
        
        for i in range(BATCH_SIZE):
            m, p, _ = calc_metrics_truncated(final_curve_real[i], x_axis)
            v = final_vol[i]
            # 计算得分 (如果体积超标则大幅惩罚)
            score = -(1.0 * m + 2.0 * p - 20.0 * v)
            if v > 0.6: score += 10000.0
            final_scores.append(score)
            
        best_idx = np.argmin(final_scores)
        best_params = final_design_phys[best_idx].cpu().numpy()
        best_curve = final_curve_real[best_idx]
        
        # 获取最佳个体的截断后属性
        b_mot, b_peak, b_disp = calc_metrics_truncated(best_curve, x_axis)
        b_vol = final_vol[best_idx]

    # --- 详细打印 ---
    print("\n" + "="*50)
    print(">>> 优化结果 (Optimized Design - Robust Metrics):")
    print("-" * 30)
    print(f"Layers (Int):       {int(best_params[0])}")
    print(f"Spacing:            {best_params[1]:.2f}")
    N_final = int(round(360.0 / best_params[2]))
    print(f"Angle (360/{N_final}):   {best_params[2]:.2f}")
    print(f"Spiral Width:       {best_params[3]:.2f}")
    print(f"Radial Width:       {best_params[4]:.2f}")
    print("-" * 30)
    print(f"MOT:  {b_mot:.4f}")
    print(f"Peak: {b_peak:.4f}")
    print(f"Disp: {b_disp:.4f} mm (Peak Displacement)")
    print(f"Vol:  {b_vol:.4f}")
    print("="*50)
    
    # 绘图 (应用截断)
    _, _, trunc_disp = calc_metrics_truncated(best_curve, x_axis)
    # 简单的截断绘图：找到对应 index
    strain_val = trunc_disp / config.LENGTH_MM
    idx_end = np.searchsorted(x_axis, strain_val) + 10 # 稍微多画一点
    idx_end = min(idx_end, len(x_axis))
    x_plot = x_axis[:idx_end]
    y_plot = best_curve[:idx_end]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x_plot, y_plot, 'g-', label='Optimized (Truncated)')
    plt.title(f"Optimized Curve (Vol={b_vol:.2f})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(perf_history)
    plt.title("Score Trend")
    plt.grid(True, alpha=0.3)
    plt.show()

# ================= 4. 反求模块 =================

def run_inversion():
    """对应代码四：曲线反求 (适配 Normalized Masked R2 + Real Physical Metrics)"""
    bundle = dataset.get_data_bundle(is_training=False)
    if not bundle: return
    
    scaler_X, scaler_Y_curve, scaler_Y_scalar = bundle['scalers']
    yc_test_raw = bundle['y_curve_test_raw']
    X_test_raw = bundle['X_test_raw']
    mask_test = bundle['mask_test']
    names_test = bundle['names_test']
    
    # [修改点] 获取动态 max_strain
    current_max_strain = bundle['max_strain']

    device = config.DEVICE
    net = model_lib.SpiderWebModel().to(device)
    try:
        net.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    except:
        print("Model not found! Please train first.")
        return
    net.eval()
    
    # MinMax Params
    x_scale = torch.FloatTensor(scaler_X.scale_).to(device)
    x_min = torch.FloatTensor(scaler_X.min_).to(device)
    ys_scale = torch.FloatTensor(scaler_Y_scalar.scale_).to(device)
    ys_min = torch.FloatTensor(scaler_Y_scalar.min_).to(device)
    
    # 随机选择样本
    # sample_idx = random.randint(0, len(names_test)-1)
    sample_idx = 0 
    real_job_name = names_test[sample_idx]
    
    # === 获取目标数据 ===
    target_curve_real = yc_test_raw[sample_idx]   # 真实物理曲线 (用于物理对比)
    target_params_gt = X_test_raw[sample_idx]     # 真实参数
    mask_s = mask_test[sample_idx]                # Mask (关键!)
    
    # 归一化目标曲线 (用于计算 Loss 和 R2)
    target_norm = scaler_Y_curve.transform(target_curve_real.reshape(1, -1))[0]
    
    # === 优化设置 ===
    BATCH_SIZE = 128
    # 扩展成 Batch 进行并行反求
    target_batch_norm = torch.FloatTensor(target_norm).to(device).unsqueeze(0).repeat(BATCH_SIZE, 1)
    mask_batch = torch.FloatTensor(mask_s).to(device).unsqueeze(0).repeat(BATCH_SIZE, 1)
    
    init_inv = torch.randn(BATCH_SIZE, 13).to(device)
    init_inv.requires_grad = True
    opt = optim.Adam([init_inv], lr=0.05)
    
    valid_n = torch.tensor([4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36], device=device).float()

    # 辅助函数
    def get_physical_params(latent_tensor):
        l_raw = torch.sigmoid(latent_tensor[:, 0]) * (config.LAYER_MAX - config.LAYER_MIN) + config.LAYER_MIN
        s_raw = torch.sigmoid(latent_tensor[:, 1]) * (config.SPACING_MAX - config.SPACING_MIN) + config.SPACING_MIN
        n_raw = torch.sigmoid(latent_tensor[:, 2]) * (config.N_RADIALS_MAX - config.N_RADIALS_MIN) + config.N_RADIALS_MIN
        angle_raw = 360.0 / n_raw
        sw_raw = torch.sigmoid(latent_tensor[:, 3]) * (config.WIDTH_MAX - config.WIDTH_MIN) + config.WIDTH_MIN
        rw_raw = torch.sigmoid(latent_tensor[:, 4]) * (config.WIDTH_MAX - config.WIDTH_MIN) + config.WIDTH_MIN
        fin_vec = torch.softmax(latent_tensor[:, 5:9], dim=1)
        fout_vec = torch.softmax(latent_tensor[:, 9:13], dim=1)
        phys = torch.cat([l_raw.unsqueeze(1), s_raw.unsqueeze(1), angle_raw.unsqueeze(1), 
                          sw_raw.unsqueeze(1), rw_raw.unsqueeze(1), fin_vec, fout_vec], dim=1)
        return phys, l_raw, n_raw

    def snap_to_valid_n_tensor(n_tensor):
        dist = torch.abs(n_tensor.unsqueeze(1) - valid_n.unsqueeze(0))
        min_indices = torch.argmin(dist, dim=1)
        return valid_n[min_indices]

    print(f">>> 开始反求样本: {real_job_name} ...")
    loss_history = []
    
    # === 优化循环 ===
    for step in range(2000):
        opt.zero_grad()
        design_phys, l_cont, n_cont = get_physical_params(init_inv)
        
        design_norm = design_phys * x_scale + x_min
        pc_norm, _ = net(design_norm)
        
        # Masked Loss (只优化有效部分)
        diff = (pc_norm - target_batch_norm) ** 2
        masked_diff = diff * mask_batch
        mse_per_sample = masked_diff.sum(dim=1) / (mask_batch.sum(dim=1) + 1e-8)
        
        int_penalty = torch.sin(l_cont * torch.pi)**2 + torch.sin(n_cont * torch.pi)**2
        loss = mse_per_sample.mean() + 2.0 * int_penalty.mean()
        loss.backward()
        opt.step()
        
        loss_history.append(mse_per_sample.mean().item())
        if (step+1) % 500 == 0:
            print(f"Step {step+1} | Curve MSE Loss: {loss.item():.6f}")

    # === 结果提取 ===
    with torch.no_grad():
        final_phys_cont, l_final_cont, n_final_cont = get_physical_params(init_inv)
        
        l_int = torch.round(l_final_cont)
        n_snapped = snap_to_valid_n_tensor(n_final_cont)
        angle_snapped = 360.0 / n_snapped
        
        final_phys_discrete = final_phys_cont.clone()
        final_phys_discrete[:, 0] = l_int
        final_phys_discrete[:, 2] = angle_snapped
        
        final_design_norm = final_phys_discrete * x_scale + x_min
        final_curve_norm, final_scalar_norm = net(final_design_norm)
        
        # 寻找 Loss 最小的个体
        diff = (final_curve_norm - target_batch_norm) ** 2
        masked_diff = diff * mask_batch
        mse_final = masked_diff.sum(dim=1) / (mask_batch.sum(dim=1) + 1e-8)
        best_idx = torch.argmin(mse_final).item()
        
        # 提取最佳结果
        best_ai_params = final_phys_discrete[best_idx].cpu().numpy()
        best_ai_curve_norm = final_curve_norm[best_idx].cpu().numpy() # 归一化的预测曲线
        
        # 提取 Volume (反归一化标量)
        final_scalar_real = final_scalar_norm * ys_scale + ys_min
        best_ai_vol = final_scalar_real[best_idx, 3].item()

        # 反归一化曲线 -> 真实物理曲线 (用于计算 MOT/Peak)
        best_ai_curve_real = scaler_Y_curve.inverse_transform(best_ai_curve_norm.reshape(1, -1))[0]

    # === R2 计算：使用归一化数据 + Mask ===
    
    # 1. 准备归一化数据
    target_norm_flat = target_norm.flatten()
    pred_norm_flat = best_ai_curve_norm.flatten()
    mask_flat = mask_s.flatten()
    
    # 2. 应用 Mask
    valid_indices = mask_flat > 0.5
    
    # 3. 计算 R2
    if valid_indices.sum() > 0:
        r2_val = r2_score(target_norm_flat[valid_indices], pred_norm_flat[valid_indices])
    else:
        r2_val = 0.0

    # === 物理属性计算 (使用真实值 + 截断函数 + Peak Disp) ===
    # [修改点] 使用 current_max_strain
    x_axis = np.linspace(0, current_max_strain, config.NUM_POINTS)
    
    # 这里直接复用全局定义的 calc_metrics_truncated
    t_mot, t_peak, t_disp = calc_metrics_truncated(target_curve_real, x_axis)
    ai_mot, ai_peak, ai_disp = calc_metrics_truncated(best_ai_curve_real, x_axis)

    # === 绘图数据准备 (仅画 Valid 部分) ===
    x_plot = x_axis[valid_indices]
    y_target_plot = target_curve_real[valid_indices]
    y_ai_plot = best_ai_curve_real[valid_indices]

    # === 详细报告打印 ===
    print("\n" + "="*60)
    print(f"【反求结果报告】 (Target ID: {real_job_name})")
    print(f" 曲线形状拟合度 (R2 Score, Normalized & Masked): {r2_val:.4f}")
    print("-" * 60)
    
    print(f"{'Metric':<15} | {'Target (Real)':<15} | {'AI Inverted':<15} | {'Error %'}")
    print("-" * 60)
    
    metrics = [
        ('MOT', t_mot, ai_mot),
        ('Peak Str', t_peak, ai_peak),
        ('Ult. Disp', t_disp, ai_disp),
        ('Volume', 'N/A', best_ai_vol) 
    ]
    
    for name, t_val, ai_val in metrics:
        if name == 'Volume':
            print(f"{name:<15} | {str(t_val):<15} | {ai_val:<15.4f} | {'-'}")
        else:
            err = abs(ai_val - t_val) / (abs(t_val) + 1e-6) * 100
            print(f"{name:<15} | {t_val:<15.4f} | {ai_val:<15.4f} | {err:<6.2f}")
            
    print("="*60)

    # 参数对比
    print(f"{'Parameter':<20} | {'True Value':<20} | {'AI Designed':<20} | {'Diff'}")
    print("-" * 75)
    param_names = ['Layers', 'Spacing', 'Angle', 'Spiral Width', 'Radial Width']
    for i in range(5):
        p_true = target_params_gt[i]
        p_ai = best_ai_params[i]
        print(f"{param_names[i]:<20} | {p_true:<20.4f} | {p_ai:<20.4f} | {abs(p_true-p_ai):.4f}")

    # === 绘图 ===
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    # 画截断后的有效部分
    plt.plot(x_plot, y_target_plot, 'k-', linewidth=2, label='Target (Real)')
    # 注意：y_ai_plot 可能长度不一致，这里为了绘图简单，画完整 masked 区域
    plt.plot(x_plot, y_ai_plot, 'r--', linewidth=2, label=f'AI Inverted')
    plt.title(f"Inversion Result: {real_job_name}")
    plt.xlabel("Strain")
    plt.ylabel("Stress (MPa)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(loss_history)
    plt.title("Inversion Optimization Loss")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(">>> 反求流程结束。")