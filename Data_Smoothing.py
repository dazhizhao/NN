import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import config  # 导入配置文件以获取路径

def plot_comparison(df_raw, df_smooth, cols_to_plot):
    """
    绘制原始数据与平滑数据的对比图，帮助用户判断平滑程度
    """
    plt.figure(figsize=(12, 6))
    
    # 获取位移列（假设第0列是位移）
    disp_col = df_raw.columns[0]
    x_axis = df_raw[disp_col].values
    
    for i, col in enumerate(cols_to_plot):
        raw_y = df_raw[col].values
        smooth_y = df_smooth[col].values
        
        # 为了绘图清晰，只取有效部分
        mask = ~np.isnan(raw_y)
        x_valid = x_axis[mask]
        raw_valid = raw_y[mask]
        smooth_valid = smooth_y[mask] # smooth 数据在无效位置也是 NaN 或未处理，需对应
        
        # 子图
        plt.subplot(1, len(cols_to_plot), i+1)
        plt.plot(x_valid, raw_valid, color='lightgray', linewidth=2, label='Raw (Noisy)')
        plt.plot(x_valid, smooth_valid, color='red', linewidth=1.5, linestyle='--', label='Smoothed')
        
        plt.title(f"Sample: {col}")
        plt.xlabel("Displacement")
        plt.ylabel("Force")
        if i == 0: plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    print(">>> 正在显示对比图，请检查平滑效果...")
    plt.show()

def smooth_and_export():
    # 1. 定义路径
    input_file = config.FILE_FORCE
    output_file = os.path.join(config.BASE_PATH, 'Force-New.xlsx')

    # ==========================================
    # [参数调整区域]
    # 建议：如果曲线依然有锯齿，将 window_length 改大 (如 51, 61)
    #       如果峰值被削平了，将 window_length 改小 (如 11, 21)
    # ==========================================
    DEFAULT_WINDOW = 31  # 窗口长度 (必须是奇数)
    POLY_ORDER = 3       # 多项式阶数 (通常不变)

    print("="*60)
    print(f"  开始执行 S-G 平滑处理 (Window={DEFAULT_WINDOW}, Poly={POLY_ORDER})")
    print("="*60)

    if not os.path.exists(input_file):
        print(f"!!! 错误: 找不到输入文件: {input_file}")
        return

    print(f">>> 正在读取原始数据: {input_file} ...")
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"!!! 读取 Excel 失败: {e}")
        return

    df_new = df.copy()
    force_cols = df.columns[1:] # 假设第0列是位移
    
    print(f">>> 检测到 {len(force_cols)} 列力数据，开始处理...")
    
    processed_count = 0
    skipped_count = 0

    for col in force_cols:
        raw_values = df[col].values
        valid_mask = ~np.isnan(raw_values)
        valid_data = raw_values[valid_mask]
        n_points = len(valid_data)
        
        if n_points > POLY_ORDER + 2:
            try:
                # 动态调整窗口
                current_window = min(DEFAULT_WINDOW, n_points)
                if current_window % 2 == 0: current_window -= 1
                if current_window <= POLY_ORDER:
                    current_window = POLY_ORDER + 2
                    if current_window % 2 == 0: current_window += 1
                
                if current_window > n_points:
                    skipped_count += 1
                    continue

                # 滤波
                smoothed_values = savgol_filter(valid_data, window_length=current_window, polyorder=POLY_ORDER)
                df_new.loc[valid_mask, col] = smoothed_values
                processed_count += 1
                
            except Exception as e:
                print(f"Warning: 列 '{col}' 处理出错: {e}")
                skipped_count += 1
        else:
            skipped_count += 1

    print("-" * 60)
    print(f">>> 处理完成: 成功 {processed_count}, 跳过 {skipped_count}")
    
    # --- 新增：随机抽取 3 列进行绘图对比 ---
    if processed_count > 0:
        sample_cols = random.sample(list(force_cols), min(3, len(force_cols)))
        plot_comparison(df, df_new, sample_cols)

    # 导出
    print(f">>> 正在保存至: {output_file} ...")
    try:
        df_new.to_excel(output_file, index=False)
        print(">>> ✅ 保存成功！")
        print("提示: 请修改 config.py -> FILE_FORCE = 'Force-New.xlsx'")
    except Exception as e:
        print(f"!!! 保存文件失败: {e}")

if __name__ == "__main__":
    smooth_and_export()