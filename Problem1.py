import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# ==============================================================================
# 1. 图表样式与字体设置 (处理中文显示问题)
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid')

# 自动寻找可用中文字体
def set_chinese_font():
    system_fonts = [f.name for f in mpl.font_manager.fontManager.ttflist]
    # 优先列表：黑体, Arial Unicode (Mac), 宋体, 微软雅黑
    preferred_fonts = ['SimHei', 'Arial Unicode MS', 'Songti SC', 'Microsoft YaHei']
    for font in preferred_fonts:
        if font in system_fonts:
            mpl.rcParams['font.sans-serif'] = [font]
            mpl.rcParams['axes.unicode_minus'] = False # 处理负号
            print(f"系统字体已设置为: {font}")
            return
    print("未找到常用中文字体，图表可能无法显示中文。")

set_chinese_font()

# ==============================================================================
# 2. 模型参数定义 (继承自模型假设)
# ==============================================================================

# --- 任务总需求 ---
M_REQ = 1.0e8  # [Tons] 1亿吨物资 (假设1)

# --- 太空电梯系统 (SES) 参数 (假设3: 独立叠加) ---
# 假设单港年通量约 17.9万吨，3个港口总计 53.7万吨
PHI_SES_YEAR = 537000.0    # [Tons/Year] 系统总年通量
COST_SES_CONSTR = 3.0e10   # [USD] 系统建设成本 (10 Billion USD)
COST_SES_VAR = 200.0      # [USD/Ton] 单位质量边际运输成本

COST_SES_FIX = 0.05 * COST_SES_CONSTR      # [USD/Year] 系统维护费

# --- 传统火箭系统 (CRL) 参数 (假设2 & 假设5) ---
LOAD_ROCKET = 125.0      # [Tons] 单枚火箭载荷 # 
N_PADS = 10              # [个] 发射台数量 (参考数据)
TURN_AROUND = 1         # [Days] 发射台周转时间 (冷却+检修) 
COST_LAUNCH_INIT = 9.7e7  # [USD] 发射成本(2022年，可回收) (97 Million USD) 
LEARNING_RATE = 1e-4        # [Float] Wright's Law 学习率 (10%) # 

# --- 权重偏好 ---
# Gamma: 成本的权重。 0.5 表示时间与成本同等重要
GAMMA_DEFAULT = 0.2


# ==============================================================================
# 3. 核心子模型函数 (对应论文公式)
# ==============================================================================

def calc_crl_capacity_rate():
    """
    计算火箭系统的最大理论年通量 (Flux Rate)
    逻辑：受限于发射台周转时间，而非火箭数量。
    公式：Phi = N_pads * (365 / tau) * m_load
    """
    flights_per_year = N_PADS * (365.0 / TURN_AROUND)
    capacity_per_year = flights_per_year * LOAD_ROCKET
    return capacity_per_year

def calc_learning_curve_cost(n_launches):
    """
    基于 Wright's Law 的积分形式计算累计成本
    公式：C_total = Integral(u0 * x^-b) dx  从 1 到 N
    """
    if n_launches <= 0:
        return 0.0
    
    # 1. 计算衰减指数 b
    # u(2n) = u(n) * (1 - xi) => 2^-b = 1 - xi => b = -log2(1 - xi)
    b = -np.log2(1 - LEARNING_RATE)
    
    # 2. 积分计算总成本 (近似求和)
    # Integral x^-b dx = (x^(1-b)) / (1-b)
    # Cost = (u0 / (1-b)) * (N^(1-b) - 1^(1-b))
    term1 = COST_LAUNCH_INIT / (1 - b)
    term2 = np.power(n_launches, 1 - b) - 1.0
    
    total_cost = term1 * term2
    return total_cost

def solve_system_metrics(alpha):
    """
    求解给定分配比例 alpha 下的系统指标
    输入: alpha (SES分配比例, 0~1)
    输出: 字典 (时间, 成本, 各分项详情)
    """
    # 1. 物资分配 (质量守恒)
    mass_ses = alpha * M_REQ
    mass_crl = (1.0 - alpha) * M_REQ

    # 2. 计算 SES 指标 (线性流)
    # T_ses = Mass / Phi
    time_ses = mass_ses / PHI_SES_YEAR if PHI_SES_YEAR > 0 else float('inf')
    # C_ses = Fix * T + Var * Mass
    cost_ses = (COST_SES_FIX * time_ses) + (COST_SES_VAR * mass_ses) # + COST_SES_CONSTR

    # 3. 计算 CRL 指标 (非线性流)
    # 3.1 确定年通量
    phi_crl = calc_crl_capacity_rate()
    # 3.2 计算时间
    time_crl = mass_crl / phi_crl if phi_crl > 0 else float('inf')
    # 3.3 计算总发射次数
    n_launches = np.ceil(mass_crl / LOAD_ROCKET)
    # 3.4 计算成本 (积分模型)
    cost_crl = calc_learning_curve_cost(n_launches)

    # 4. 系统耦合 (并行短板效应)
    # 总时间取决于最慢的那个系统
    time_total = max(time_ses, time_crl)

    # 总成本是两者之和
    cost_total = cost_ses + cost_crl

    return {
        "Alpha": alpha,
        "Time_Total": time_total,
        "Cost_Total": cost_total,
        "Time_SES": time_ses,
        "Time_CRL": time_crl,
        "Cost_SES": cost_ses,
        "Cost_CRL": cost_crl,
        "N_Launches": n_launches
    }

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def normalize_metrics(df):
    """
    对时间和成本进行归一化处理，便于综合评价，使用多种方法进行归一化
    """

    alphas = np.linspace(0, 1, 101)
    results = [solve_system_metrics(a) for a in alphas]
    df = pd.DataFrame(results)

    # Raw Data
    raw_time = df['Time_Total'].values.reshape(-1, 1)
    raw_cost = df['Cost_Total'].values.reshape(-1, 1)

    # Normalization Methods
    methods = {
        # 'MinMax': MinMaxScaler(),
        'ZScore': StandardScaler(),
        # 'Robust': RobustScaler()
    }

    normalized_cache = {}
    for name, scaler in methods.items():
        norm_time = scaler.fit_transform(raw_time).flatten()
        norm_cost = scaler.fit_transform(raw_cost).flatten()
        normalized_cache[name] = {
            'Time': norm_time,
            'Cost': norm_cost
        }

    return df, normalized_cache

def find_optimal_elevator_percentage(gamma=GAMMA_DEFAULT):

    print(f"Learning Rate: {LEARNING_RATE*100}% (Wright's Law)")
    # Scan Elevator Percentage
    elevator_percentage_list = np.linspace(0, 1, 101)
    results = [solve_system_metrics(a) for a in elevator_percentage_list]
    df = pd.DataFrame(results)

    # Normalize Metrics
    df, normalized_cache = normalize_metrics(df)
    df['J_Score'] =  gamma * normalized_cache['ZScore']['Cost'] + (1 - gamma) * normalized_cache['ZScore']['Time']

    # Find Key Solutions

    # Optimal Solution
    idx_opt = df['J_Score'].idxmin()
    opt_sol = df.iloc[idx_opt]
    print("\n=== Optimal Solution ===")
    print(opt_sol)

    # Time Min Solution
    idx_time_min = df['Time_Total'].idxmin()
    time_min_sol = df.iloc[idx_time_min]
    print("\n=== Time Min Solution ===")
    print(time_min_sol)

    # Cost Min Solution
    idx_cost_min = df['Cost_Total'].idxmin()
    cost_min_sol = df.iloc[idx_cost_min]
    print("\n=== Cost Min Solution ===")
    print(cost_min_sol)

    # Summary
    print("\n=== Summary of Key Solutions ===")
    print(f"1. Pure Rocket, with Alpha=0")
    print(f"Time: {df.iloc[0]['Time_Total']:.2f} years, Cost: ${df.iloc[0]['Cost_Total']/1e9:.2f} Billion")

    print(f"2. Pure Elevator, with Alpha=1")
    print(f"Time: {df.iloc[-1]['Time_Total']:.2f} years, Cost: ${df.iloc[-1]['Cost_Total']/1e9:.2f} Billion")

    print(f"3. Optimal Solution at Alpha={opt_sol['Alpha']:.4f}")
    print(f"Time: {opt_sol['Time_Total']:.2f} years, Cost: ${opt_sol['Cost_Total']/1e9:.2f} Billion")
    print(f"Number of Launches: {int(opt_sol['N_Launches'])}")

    df.to_csv('Problem1_Model_Results.csv', index=False)
    print("数据已保存至: Problem1_Model_Results.csv")

def find_optimal_gamma(alpha=0.54):
    
    gamma_values = np.linspace(0, 1, 101)
    results = []
    for gamma in gamma_values:
        # Normalize Metrics
        df, normalized_cache = normalize_metrics(pd.DataFrame([solve_system_metrics(alpha) for alpha in np.linspace(0, 1, 101)]))
        J_Score = gamma * normalized_cache['ZScore']['Cost'] + (1 - gamma) * normalized_cache['ZScore']['Time']
        idx_opt = J_Score.argmin()
        opt_sol = df.iloc[idx_opt]
        results.append({
            "Gamma": gamma,
            "Alpha_Optimal": opt_sol['Alpha'],
            "Time_Total": opt_sol['Time_Total'],
            "Cost_Total": opt_sol['Cost_Total']
        })
    df_gamma = pd.DataFrame(results)
    print("\n=== Optimal Gamma Analysis ===")
    print(df_gamma)
    df_gamma.to_csv('Problem1_Optimal_Gamma_Analysis.csv', index=False)
    print("数据已保存至: Problem1_Optimal_Gamma_Analysis.csv")

def model1():
    find_optimal_elevator_percentage(GAMMA_DEFAULT)
    find_optimal_gamma(alpha=0.54)

if __name__ == "__main__":
    model1()
