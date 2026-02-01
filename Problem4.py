import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from dataclasses import dataclass

# ==============================================================================
# 1. 图表样式与字体设置 (处理中文显示问题)
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid')


def set_chinese_font():
    system_fonts = [f.name for f in mpl.font_manager.fontManager.ttflist]
    preferred_fonts = ['SimHei', 'Arial Unicode MS', 'Songti SC', 'Microsoft YaHei']
    for font in preferred_fonts:
        if font in system_fonts:
            mpl.rcParams['font.sans-serif'] = [font]
            mpl.rcParams['axes.unicode_minus'] = False
            print(f"系统字体已设置为: {font}")
            return
    print("未找到常用中文字体，图表可能无法显示中文。")


set_chinese_font()

# ==============================================================================
# 2. 模型参数定义 (Problem1 + Problem2 不完美参数)
# ==============================================================================

# --- 任务总需求 ---
M_REQ = 1.0e8  # [Tons] 1亿吨物资

# --- 太空电梯系统 (SES) 参数 ---
PHI_SES_YEAR = 537000.0    # [Tons/Year] 系统总年通量
COST_SES_CONSTR = 3.0e10   # [USD] 系统建设成本
COST_SES_VAR = 200.0       # [USD/Ton] 单位质量边际运输成本
COST_SES_FIX = 0.05 * COST_SES_CONSTR      # [USD/Year] 系统维护费
N_ELEVATORS = 3

# --- 传统火箭系统 (CRL) 参数 ---
LOAD_ROCKET = 125.0      # [Tons] 单枚火箭载荷 (Problem4 假设)
N_PADS = 10              # [个] 发射台数量
TURN_AROUND = 1          # [Days] 发射台周转时间
COST_LAUNCH_INIT = 9.7e7  # [USD] 发射成本(2022年，可回收)
LEARNING_RATE = 1e-4     # [Float] Wright's Law 学习率

# --- 不完美运行 (来自 Problem2, 使用均值) ---
SWAY_LOSS_MEAN = 0.10
ELEVATOR_FAILURE_RATE = 1.0  # 次/年/电梯
ELEVATOR_DOWNTIME_DAYS = 14  # 天

ROCKET_FAIL_RATE = 0.02
ROCKET_LAUNCH_PROB = 0.95

# --- 权重偏好 ---
GAMMA_DEFAULT = 0.2

# ==============================================================================
# 3. 火箭燃料与环境成本 (Problem4)
# ==============================================================================

RAW_SITES = [
    ("France Guyana", 5.2),
    ("India", 13.7),
    ("Texas", 26.0),
    ("Florida", 28.5),
    ("California", 34.6),
    ("Virginia", 37.9),
    ("Taiyuan", 38.8),
    ("New Zealand", 39.3),
    ("Kazakhstan", 46.0),
    ("Alaska", 57.4),
]


@dataclass(frozen=True)
class RocketParams:
    G0: float = 9.80665  # [m/s^2]
    ISP: float = 450.0   # [s]
    DV_TLI_BASE: float = 9400.0  # [m/s] 基础速度需求
    V_PARK: float = 7800.0       # [m/s] 近地停泊轨道速度
    MOON_INC: float = 28.5       # [deg] 目标轨道倾角


PARAMS = RocketParams()

# 结构比 & 多级火箭限制
STRUCTURAL_FRACTION = 0.08
MAX_STAGE_RATIO = 10.0

# 环境净化成本 (USD/ton fuel)
ENV_CLEANUP_COST_PER_TON = 100.0


def calc_fuel_mass_for_lat(lat_deg, payload_tons):
    """
    计算特定纬度的燃料需求 (吨)
    包含 Model 4.4.1 的自转损失和平面机动惩罚。
    包含多级火箭逻辑以避免 inf。
    """
    phi = abs(lat_deg)
    phi_rad = np.radians(phi)

    # 1. 自转损失
    dv_rot = 465.0 * (1.0 - np.cos(phi_rad))

    # 2. 平面机动 (Plane Change)
    dv_inc = 0.0
    if phi > PARAMS.MOON_INC:
        delta = np.radians(phi - PARAMS.MOON_INC)
        dv_inc = 2 * PARAMS.V_PARK * np.sin(delta / 2.0)

    dv_total = PARAMS.DV_TLI_BASE + dv_rot + dv_inc
    ve = PARAMS.ISP * PARAMS.G0

    total_ratio = np.exp(dv_total / ve)
    if total_ratio <= MAX_STAGE_RATIO:
        stage_dvs = [dv_total]
    else:
        n_stages = int(np.ceil(np.log(total_ratio) / np.log(MAX_STAGE_RATIO)))
        stage_dvs = [dv_total / n_stages] * n_stages

    payload_mass = payload_tons
    total_propellant = 0.0

    for stage_dv in stage_dvs:
        stage_ratio = np.exp(stage_dv / ve)
        epsilon = STRUCTURAL_FRACTION
        if stage_ratio * epsilon >= 0.95:
            epsilon = 0.95 / stage_ratio

        denom = stage_ratio * epsilon - 1.0
        if denom >= -1e-6:
            raise ValueError("结构比过大导致不可行的质量比，请调整参数。")

        stage_mass = payload_mass * (1.0 - stage_ratio) / denom
        prop_mass = stage_mass * (1.0 - epsilon)
        total_propellant += prop_mass
        payload_mass += stage_mass

    return total_propellant, dv_total


def optimize_launch_allocation(total_launches):
    fuel_per_launch = []
    for _, lat in RAW_SITES:
        fuel_mass, _ = calc_fuel_mass_for_lat(lat, LOAD_ROCKET)
        fuel_per_launch.append(fuel_mass)

    weights = np.array([1.0 / max(1e-9, fuel) for fuel in fuel_per_launch])
    weights = weights / weights.sum()

    raw_allocations = total_launches * weights
    launches_int = np.floor(raw_allocations).astype(int)
    remainder = total_launches - launches_int.sum()

    if remainder > 0:
        fractional = raw_allocations - launches_int
        order = np.argsort(-fractional)
        for idx in order[:remainder]:
            launches_int[idx] += 1

    return launches_int, weights, fuel_per_launch


def calc_site_impacts(total_launches):
    site_records = []
    if total_launches <= 0:
        for name, lat in RAW_SITES:
            site_records.append({
                "Site": name,
                "Latitude": lat,
                "Launches": 0,
                "Allocation_Ratio": 0.0,
                "Fuel_Per_Launch_Tons": 0.0,
                "Fuel_Total_Tons": 0.0,
                "Env_Cost": 0.0,
            })
        return pd.DataFrame(site_records)

    launches_alloc, weights, fuel_per_launch_list = optimize_launch_allocation(total_launches)

    for idx, (name, lat) in enumerate(RAW_SITES):
        launches = int(launches_alloc[idx])
        fuel_per_launch = fuel_per_launch_list[idx]
        _, dv_total = calc_fuel_mass_for_lat(lat, LOAD_ROCKET)
        fuel_total = fuel_per_launch * launches
        env_cost = fuel_total * ENV_CLEANUP_COST_PER_TON
        site_records.append({
            "Site": name,
            "Latitude": lat,
            "Launches": launches,
            "Allocation_Ratio": weights[idx],
            "Fuel_Per_Launch_Tons": fuel_per_launch,
            "Fuel_Total_Tons": fuel_total,
            "Env_Cost": env_cost,
            "DeltaV_Total": dv_total,
        })

    return pd.DataFrame(site_records)


# ==============================================================================
# 4. 核心模型函数
# ==============================================================================


def calc_crl_capacity_rate():
    flights_per_year = N_PADS * (365.0 / TURN_AROUND)
    return flights_per_year * LOAD_ROCKET


def calc_learning_curve_cost(n_launches):
    if n_launches <= 0:
        return 0.0
    b = -np.log2(1 - LEARNING_RATE)
    term1 = COST_LAUNCH_INIT / (1 - b)
    term2 = np.power(n_launches, 1 - b) - 1.0
    return term1 * term2


def calc_ses_time_cost(mass_ses):
    base_time_ses = mass_ses / PHI_SES_YEAR if PHI_SES_YEAR > 0 else float('inf')
    loss_factor = 1.0 / max(1e-6, (1.0 - SWAY_LOSS_MEAN))

    expected_uptime = 1.0 - (ELEVATOR_FAILURE_RATE * ELEVATOR_DOWNTIME_DAYS / 365.0)
    capacity_factor = max(1e-6, expected_uptime)

    time_ses = base_time_ses * loss_factor / capacity_factor
    cost_ses = (COST_SES_FIX * time_ses) + (COST_SES_VAR * mass_ses)
    return time_ses, cost_ses


def calc_crl_time_cost(mass_crl):
    if mass_crl <= 0:
        return 0.0, 0.0, 0.0, 0

    required_success = int(np.ceil(mass_crl / LOAD_ROCKET))
    expected_total_launches = required_success / max(1e-6, (1.0 - ROCKET_FAIL_RATE))
    flights_per_year = N_PADS * (365.0 / TURN_AROUND) * ROCKET_LAUNCH_PROB
    time_crl = expected_total_launches / flights_per_year
    cost_crl = calc_learning_curve_cost(expected_total_launches)
    return time_crl, cost_crl, expected_total_launches, required_success


def solve_system_metrics(alpha):
    mass_ses = alpha * M_REQ
    mass_crl = (1.0 - alpha) * M_REQ

    time_ses, cost_ses = calc_ses_time_cost(mass_ses)
    time_crl, cost_crl, expected_launches, required_success = calc_crl_time_cost(mass_crl)

    total_launches_int = int(np.ceil(expected_launches))
    site_df = calc_site_impacts(total_launches_int)
    env_cost = site_df["Env_Cost"].sum()

    time_total = max(time_ses, time_crl)
    cost_total = cost_ses + cost_crl + env_cost

    return {
        "Alpha": alpha,
        "Time_Total": time_total,
        "Cost_Total": cost_total,
        "Time_SES": time_ses,
        "Time_CRL": time_crl,
        "Cost_SES": cost_ses,
        "Cost_CRL": cost_crl,
        "Cost_Env": env_cost,
        "N_Launches_Expected": expected_launches,
        "N_Launches_Required": required_success,
    }, site_df


def normalize_metrics(df):
    raw_time = df['Time_Total'].values.reshape(-1, 1)
    raw_cost = df['Cost_Total'].values.reshape(-1, 1)

    time_mean = raw_time.mean()
    time_std = raw_time.std(ddof=0)
    cost_mean = raw_cost.mean()
    cost_std = raw_cost.std(ddof=0)

    norm_time = (raw_time - time_mean) / time_std
    norm_cost = (raw_cost - cost_mean) / cost_std

    return norm_time.flatten(), norm_cost.flatten()


def find_optimal_alpha(df, gamma=GAMMA_DEFAULT):
    norm_time, norm_cost = normalize_metrics(df)
    df = df.copy()
    df['J_Score'] = gamma * norm_cost + (1.0 - gamma) * norm_time

    idx_opt = df['J_Score'].idxmin()
    return df, df.loc[idx_opt]


def model4():
    alpha_values = np.linspace(0, 1, 101)

    results = []
    site_tables = []
    for alpha in alpha_values:
        metrics, site_df = solve_system_metrics(alpha)
        results.append(metrics)
        site_df = site_df.copy()
        site_df["Alpha"] = alpha
        site_tables.append(site_df)

    df = pd.DataFrame(results)
    df, opt_sol = find_optimal_alpha(df, GAMMA_DEFAULT)

    idx_time_min = df['Time_Total'].idxmin()
    idx_cost_min = df['Cost_Total'].idxmin()

    print("\n=== Optimal Solution ===")
    print(opt_sol)
    print("\n=== Time Min Solution ===")
    print(df.loc[idx_time_min])
    print("\n=== Cost Min Solution ===")
    print(df.loc[idx_cost_min])

    df.to_csv('Problem4_Model_Results.csv', index=False)
    site_summary = pd.concat(site_tables, ignore_index=True)
    site_summary.to_csv('Problem4_Site_Impacts.csv', index=False)

    print("数据已保存至: Problem4_Model_Results.csv")
    print("数据已保存至: Problem4_Site_Impacts.csv")


if __name__ == "__main__":
    model4()
