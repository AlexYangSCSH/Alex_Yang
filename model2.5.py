import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

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
# 2. 模型参数定义
# ==============================================================================

# --- 任务总需求 ---
M_REQ = 1.0e8  # [Tons]

# --- 太空电梯系统 (SES) ---
PHI_SES_YEAR = 537000.0    # [Tons/Year]
COST_SES_CONSTR = 3.0e10   # [USD]
COST_SES_VAR = 200.0       # [USD/Ton]
COST_SES_FIX = 0.05 * COST_SES_CONSTR  # [USD/Year]
NUM_ELEVATORS = 3

# --- 火箭系统 (CRL) 基线 ---
LOAD_ROCKET_INIT = 125.0        # [Tons]
LOAD_ROCKET_MAX = 250.0         # [Tons]
N_PADS_INIT = 10
N_PADS_MAX = 100
FREQ_PER_DAY_INIT = 1.0 / 3.0   # [launches/day]
FREQ_PER_DAY_MAX = 3.0
TURN_AROUND_DAYS = 1.0
COST_LAUNCH_INIT = 9.7e7         # [USD]
LEARNING_RATE = 0.1
COST_LAUNCH_FLOOR = 5e6         # [USD] 第一性原理成本底线

# --- Logistic 增长参数 ---
LOGISTIC_K_PADS = 0.08
LOGISTIC_T0_PADS = 20
LOGISTIC_K_PAYLOAD = 0.06
LOGISTIC_T0_PAYLOAD = 18
LOGISTIC_K_FREQ = 0.07
LOGISTIC_T0_FREQ = 15

# --- Monte Carlo 参数 ---
RNG_SEED = 42
N_SIM = 1000

# --- NPV 经济参数 ---
DISCOUNT_RATE = 0.05
OPPORTUNITY_COST = 3.0e11  # [USD/Year] 3000亿美元

# --- 动态决策权重 (全局默认值) ---
WEIGHT_COST = 0.5
WEIGHT_TIME = 0.5

# --- 非完美运行扰动 ---
ELEVATOR_SWAY_MEAN = 0.10
ELEVATOR_SWAY_STD = 0.05
ELEVATOR_FAILURE_RATE = 1.0   # [1/year]
ELEVATOR_REPAIR_DAYS = 14

ROCKET_FAILURE_RATE = 0.02
ROCKET_WEATHER_PROB = 0.95
ROCKET_WINDOW_PROB_RANGE = (0.20, 0.25)


# ==============================================================================
# 3. 核心子模型函数
# ==============================================================================


def logistic_growth(t, lower, upper, k, t0):
    return lower + (upper - lower) / (1 + np.exp(-k * (t - t0)))


def calc_learning_curve_cost_total(n_launches):
    if n_launches <= 0:
        return 0.0
    b = -np.log2(1 - LEARNING_RATE)
    term1 = COST_LAUNCH_INIT / (1 - b)
    term2 = np.power(n_launches, 1 - b) - 1.0
    return term1 * term2

def calc_learning_curve_cost_total_with_floor(n_launches):
    if n_launches <= 0:
        return 0.0
    
    # 计算衰减指数 b
    # 如果 LEARNING_RATE 太小，b 会趋近于 0，导致后面计算溢出
    if LEARNING_RATE < 1e-3: 
        # 如果学习率极低，近似认为没有学习效应，直接按初始价格算（或者仅按线性算）
        return n_launches * COST_LAUNCH_INIT

    b = -np.log2(1 - LEARNING_RATE)
    
    # 增加溢出保护
    try:
        ratio = COST_LAUNCH_INIT / COST_LAUNCH_FLOOR
        exponent = 1 / b
        
        # 预判：如果底数 > 1 且指数极大，直接认为阈值是无穷大
        # 194 ** 100 已经非常大了，这里设置一个安全上限
        if np.log10(ratio) * exponent > 300: 
            threshold = float('inf')
        else:
            threshold = ratio ** exponent
            
    except (OverflowError, RuntimeWarning):
        threshold = float('inf')

    # 处理无穷大情况：如果阈值无限大，说明在仿真周期内永远降不到底板价
    if threshold == float('inf') or threshold > 1e15:
        return calc_learning_curve_cost_total(n_launches)

    threshold_int = max(1, int(np.ceil(threshold)))

    if n_launches <= threshold_int:
        return calc_learning_curve_cost_total(n_launches)

    cost_before_floor = calc_learning_curve_cost_total(threshold_int)
    cost_after_floor = COST_LAUNCH_FLOOR * (n_launches - threshold_int)
    return cost_before_floor + cost_after_floor


def sample_truncated_normal(mean, std, lower=0.0, upper=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    value = rng.normal(mean, std)
    while value < lower or value > upper:
        value = rng.normal(mean, std)
    return value


def simulate_yearly_capacity(year_idx, rng):
    pads = logistic_growth(year_idx, N_PADS_INIT, N_PADS_MAX, LOGISTIC_K_PADS, LOGISTIC_T0_PADS)
    payload = logistic_growth(year_idx, LOAD_ROCKET_INIT, LOAD_ROCKET_MAX, LOGISTIC_K_PAYLOAD, LOGISTIC_T0_PAYLOAD)
    freq = logistic_growth(year_idx, FREQ_PER_DAY_INIT, FREQ_PER_DAY_MAX, LOGISTIC_K_FREQ, LOGISTIC_T0_FREQ)

    window_prob = rng.uniform(*ROCKET_WINDOW_PROB_RANGE)
    launch_day_prob = ROCKET_WEATHER_PROB * window_prob
    available_days = 365.0 * launch_day_prob

    attempts_capacity = pads * freq * available_days
    return {
        "Pads": pads,
        "Payload": payload,
        "Frequency": freq,
        "Attempts_Capacity": attempts_capacity,
    }


def simulate_elevator_capacity(rng):
    sway_loss = sample_truncated_normal(
        ELEVATOR_SWAY_MEAN,
        ELEVATOR_SWAY_STD,
        lower=0.0,
        upper=1.0,
        rng=rng,
    )
    availabilities = []
    for _ in range(NUM_ELEVATORS):
        failures = rng.poisson(ELEVATOR_FAILURE_RATE)
        downtime_fraction = (failures * ELEVATOR_REPAIR_DAYS) / 365.0
        availability = max(0.0, 1.0 - downtime_fraction)
        availabilities.append(availability)
    availability_factor = np.mean(availabilities)
    effective_phi = PHI_SES_YEAR * (1.0 - sway_loss) * availability_factor
    return effective_phi


def simulate_year(
    remaining_mass,
    year_idx,
    rng,
    mode="dynamic",
    cumulative_attempts=0,
    weight_cost=WEIGHT_COST,
    weight_time=WEIGHT_TIME,
):
    elevator_capacity = simulate_elevator_capacity(rng)
    rocket_state = simulate_yearly_capacity(year_idx, rng)

    rocket_payload = rocket_state["Payload"]
    rocket_attempts = int(np.floor(rocket_state["Attempts_Capacity"]))
    rocket_expected_payload = rocket_payload * (1.0 - ROCKET_FAILURE_RATE)

    # 1. 计算火箭单价 ($/Ton)
    rocket_cost_per_launch = 0.0
    if rocket_attempts > 0:
        cost_total_new = calc_learning_curve_cost_total_with_floor(cumulative_attempts + 1)
        cost_total_prev = calc_learning_curve_cost_total_with_floor(cumulative_attempts)
        rocket_cost_per_launch = cost_total_new - cost_total_prev

    if rocket_expected_payload > 0:
        rocket_cost_per_ton = rocket_cost_per_launch / rocket_expected_payload
    else:
        rocket_cost_per_ton = float("inf")

    # ==============================================================================
    # 2. 【核心修正】计算时间价值 ($/Ton)
    # ==============================================================================
    # 原逻辑：OPPORTUNITY_COST / remaining_mass (分母太大，导致惩罚被稀释)
    # 新逻辑：OPPORTUNITY_COST / 期望年运量
    # 物理含义：这1000亿的损失，是由这一年本该完成的运输量（比如50万吨）分摊的。
    # 如果不运这1吨，它卡住的是这一年的流水线。
    
    BENCHMARK_CAPACITY = 500000.0 # [Tons] 设定为电梯的基准年运力
    opportunity_cost_per_ton = OPPORTUNITY_COST / BENCHMARK_CAPACITY
    
    # 这样算出来：1e11 / 5e5 = 200,000 USD/Ton。
    # 配合 Urgent 权重 (0.2 vs 0.8):
    # 火箭代价: 0.2 * 1,000,000 = 200,000
    # 等待代价: 0.8 * 200,000 = 160,000
    # 差距瞬间缩小！随着火箭稍微降价一点点，马上就会触发发射！
    # ==============================================================================

    rocket_allowed = (
        weight_cost * rocket_cost_per_ton < weight_time * opportunity_cost_per_ton
    )

    if mode == "elevator_only":
        rocket_capacity = 0.0
        rocket_attempts = 0
    elif mode == "rocket_only":
        # 纯火箭模式：强制无视成本，火力全开
        rocket_capacity = rocket_payload * rocket_attempts * (1.0 - ROCKET_FAILURE_RATE)
        elevator_capacity = 0.0
    else:
        if rocket_allowed:
            rocket_capacity = rocket_payload * rocket_attempts * (1.0 - ROCKET_FAILURE_RATE)
        else:
            rocket_capacity = 0.0
            rocket_attempts = 0

    total_capacity = elevator_capacity + rocket_capacity
    
    # ... (以下代码保持不变) ...
    if total_capacity <= 0:
        return {
            "Delivered": 0.0,
            "Elevator_Delivered": 0.0,
            "Rocket_Delivered": 0.0,
            "Rocket_Attempts": 0,
            "Time_Fraction": 1.0,
            "Cost_Ses": 0.0,
            "Cost_Rocket": 0.0,
        }

    if remaining_mass >= total_capacity:
        fraction_year = 1.0
        elevator_delivered = elevator_capacity
        rocket_delivered = rocket_capacity
    else:
        fraction_year = remaining_mass / total_capacity
        elevator_delivered = elevator_capacity * fraction_year
        rocket_delivered = rocket_capacity * fraction_year

    attempts_this_year = int(np.floor(rocket_attempts * fraction_year))
    successes = rng.binomial(attempts_this_year, 1.0 - ROCKET_FAILURE_RATE)
    rocket_delivered = successes * rocket_payload

    time_fraction = fraction_year
    cost_ses = (COST_SES_FIX * time_fraction) + (COST_SES_VAR * elevator_delivered)

    return {
        "Delivered": elevator_delivered + rocket_delivered,
        "Elevator_Delivered": elevator_delivered,
        "Rocket_Delivered": rocket_delivered,
        "Rocket_Attempts": attempts_this_year,
        "Time_Fraction": time_fraction,
        "Cost_Ses": cost_ses,
        "Cost_Rocket": 0.0,
    }


# ==============================================================================
# 4. 蒙特卡洛仿真
# ==============================================================================


def simulate_scenario(mode, rng, weight_cost=WEIGHT_COST, weight_time=WEIGHT_TIME):
    remaining_mass = M_REQ
    year_idx = 0
    time_elapsed = 0.0
    cumulative_elevator = 0.0
    cumulative_rocket = 0.0
    cumulative_attempts = 0
    total_npv_cost = 0.0

    while remaining_mass > 0 and year_idx < 2000:
        year_data = simulate_year(
            remaining_mass,
            year_idx,
            rng,
            mode=mode,
            cumulative_attempts=cumulative_attempts,
            weight_cost=weight_cost,
            weight_time=weight_time,
        )
        delivered = year_data["Delivered"]
        remaining_mass = max(0.0, remaining_mass - delivered)
        time_elapsed += year_data["Time_Fraction"]
        cumulative_elevator += year_data["Elevator_Delivered"]
        cumulative_rocket += year_data["Rocket_Delivered"]

        attempts_prev = cumulative_attempts
        cumulative_attempts += year_data["Rocket_Attempts"]
        rocket_cost = 0.0
        if cumulative_attempts > 0:
            cost_total_new = calc_learning_curve_cost_total_with_floor(cumulative_attempts)
            cost_total_prev = calc_learning_curve_cost_total_with_floor(attempts_prev)
            rocket_cost = cost_total_new - cost_total_prev
        ses_cost = year_data["Cost_Ses"]
        year_operation_cost = rocket_cost + ses_cost
        year_opportunity_cost = OPPORTUNITY_COST * year_data["Time_Fraction"]
        discount_factor = (1.0 + DISCOUNT_RATE) ** (year_idx + 1)
        total_npv_cost += (year_operation_cost + year_opportunity_cost) / discount_factor

        year_idx += 1

    effective_alpha = cumulative_elevator / M_REQ
    return {
        "Time_Total": time_elapsed,
        "NPV_Total_Cost": total_npv_cost,
        "Effective_Alpha": effective_alpha,
        "Rocket_Attempts": cumulative_attempts,
    }


def run_monte_carlo():
    rng = np.random.default_rng(RNG_SEED)
    scenarios = ["dynamic", "elevator_only", "rocket_only"]
    weight_configs = [
        {"Label": "Balanced", "Weight_Cost": 0.5, "Weight_Time": 0.5},
        {"Label": "Cost_Saving", "Weight_Cost": 0.8, "Weight_Time": 0.2},
        {"Label": "Urgent", "Weight_Cost": 0.2, "Weight_Time": 0.8},
    ]
    samples = []
    
    print(f"开始蒙特卡洛模拟 (N_SIM={N_SIM})...")
    
    for weight_config in weight_configs:
        weight_label = weight_config["Label"]
        weight_cost = weight_config["Weight_Cost"]
        weight_time = weight_config["Weight_Time"]
        for scenario in scenarios:
            print(f"--> 正在模拟场景: {scenario} | 权重: {weight_label} ...")
            for sim_idx in range(N_SIM):
                # 每 10 次打印一个点，或者显示进度
                if sim_idx % 10 == 0:
                    print(f"    进度: {sim_idx}/{N_SIM}", end="\r")
                    
                metrics = simulate_scenario(
                    scenario,
                    rng,
                    weight_cost=weight_cost,
                    weight_time=weight_time,
                )
                metrics["Scenario"] = scenario
                metrics["Weight_Group"] = weight_label
                metrics["Weight_Cost"] = weight_cost
                metrics["Weight_Time"] = weight_time
                metrics["Sim"] = sim_idx
                samples.append(metrics)
            print(f"    进度: {N_SIM}/{N_SIM} [完成]") 
        
    return pd.DataFrame(samples)


def summarize_distribution(df, scenario, weight_group):
    subset = df[(df["Scenario"] == scenario) & (df["Weight_Group"] == weight_group)]
    summary = {}
    for metric in ["Time_Total", "NPV_Total_Cost", "Effective_Alpha"]:
        values = subset[metric].values
        mean_val = np.mean(values)
        var_val = np.var(values, ddof=0)
        ci_low = np.percentile(values, 2.5)
        ci_high = np.percentile(values, 97.5)
        summary[f"{metric}_Mean"] = mean_val
        summary[f"{metric}_Var"] = var_val
        summary[f"{metric}_CI_Low"] = ci_low
        summary[f"{metric}_CI_High"] = ci_high
    return summary


def problem2():
    df_samples = run_monte_carlo()

    summaries = []
    for weight_group in df_samples["Weight_Group"].unique():
        for scenario in ["dynamic", "elevator_only", "rocket_only"]:
            summary = summarize_distribution(df_samples, scenario, weight_group)
            summary["Scenario"] = scenario
            summary["Weight_Group"] = weight_group
            summaries.append(summary)

    df_summary = pd.DataFrame(summaries)

    print("\n=== Monte Carlo Summary ===")
    print(df_summary)

    print("\n=== 三种方案的成本对比简报 (NPV) ===")
    brief_rows = []
    for weight_group in ["Balanced", "Cost_Saving", "Urgent"]:
        for scenario in ["dynamic", "elevator_only", "rocket_only"]:
            subset = df_samples[
                (df_samples["Scenario"] == scenario)
                & (df_samples["Weight_Group"] == weight_group)
            ]
            mean_cost = subset["NPV_Total_Cost"].mean() / 1e9
            mean_time = subset["Time_Total"].mean()
            mean_alpha = subset["Effective_Alpha"].mean()
            brief_rows.append(
                {
                    "Weight_Group": weight_group,
                    "Scenario": scenario,
                    "Mean_Total_Time": f"{mean_time:.1f}",
                    "NPV_Total_Cost (Billion USD)": f"{mean_cost:.2f}",
                    "Mean_Effective_Alpha": f"{mean_alpha:.3f}",
                }
            )
    print(pd.DataFrame(brief_rows).to_string(index=False))

    df_samples.to_csv("Problem2_MonteCarlo_Samples.csv", index=False)
    df_summary.to_csv("Problem2_MonteCarlo_Summary.csv", index=False)
    print("数据已保存至: Problem2_MonteCarlo_Samples.csv, Problem2_MonteCarlo_Summary.csv")


if __name__ == "__main__":
    problem2()