import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ==============================================================================
# 1. 图表样式与字体设置 (处理中文显示问题)
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid')


def set_chinese_font():
    system_fonts = [f.name for f in mpl.font_manager.fontManager.ttflist]
    preferred_fonts = ['SimHei', 'Arial Unicode MS', 'Songti SC', 'M icrosoft YaHei']
    for font in preferred_fonts:
        if font in system_fonts:
            mpl.rcParams['font.sans-serif'] = [font]
            mpl.rcParams['axes.unicode_minus'] = False
            print(f"系统字体已设置为: {font}")
            return
    print("未找到常用中文字体，图表可能无法显示中文。")


set_chinese_font()

# ==============================================================================
# 2. 模型参数定义 (继承自模型假设)
# ==============================================================================

M_REQ = 8300100  * 0.05 # [Tons] 每年需要运送的水量，假设95%的回收率

# --- 太空电梯系统 (SES) 参数 ---
PHI_SES_YEAR = 537000.0    # [Tons/Year] 系统总年通量
COST_SES_CONSTR = 3.0e10   # [USD] 系统建设成本 (10 Billion USD)
COST_SES_VAR = 200.0       # [USD/Ton] 单位质量边际运输成本
COST_SES_FIX = 0.05 * COST_SES_CONSTR      # [USD/Year] 系统维护费
N_ELEVATORS = 3

# --- 传统火箭系统 (CRL) 参数 ---
LOAD_ROCKET = 125.0      # [Tons] 单枚火箭载荷
N_PADS = 10              # [个] 发射台数量
TURN_AROUND = 1          # [Days] 发射台周转时间
COST_LAUNCH_INIT = 9.7e7  # [USD] 发射成本(2022年，可回收)
LEARNING_RATE = 1e-4     # [Float] Wright's Law 学习率

# --- 权重偏好 ---
GAMMA_DEFAULT = 0.2

# ==============================================================================
# 3. 不确定性假设 (Monte Carlo)
# ==============================================================================

RNG_SEED = 42
N_SIM = 1000

# 太空电梯绳摇晃损失 (截断正态)
SWAY_LOSS_MEAN = 0.10
SWAY_LOSS_VAR = 0.05
SWAY_LOSS_STD = np.sqrt(SWAY_LOSS_VAR)
SWAY_LOSS_LOWER = 0.0
SWAY_LOSS_UPPER = 1.0

# 太空电梯故障
ELEVATOR_FAILURE_RATE = 1.0  # 次/年/电梯
ELEVATOR_DOWNTIME_DAYS = 14  # 天

# 火箭系统不确定性
ROCKET_FAIL_RATE = 0.02
ROCKET_LAUNCH_PROB = 0.95


# ==============================================================================
# 4. 核心子模型函数
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


def sample_truncated_normal(rng, mean, std, lower, upper, size=1):
    samples = []
    while len(samples) < size:
        draw = rng.normal(mean, std, size - len(samples))
        for value in draw:
            if lower <= value <= upper:
                samples.append(value)
    return np.array(samples)


def simulate_system_metrics(alpha, rng):
    mass_ses = alpha * M_REQ
    mass_crl = (1.0 - alpha) * M_REQ

    # SES: 基础时间
    base_time_ses = mass_ses / PHI_SES_YEAR if PHI_SES_YEAR > 0 else float('inf')

    # SES: 绳摇晃损失 (损失越大，等效时间越长)
    sway_loss = sample_truncated_normal(
        rng, SWAY_LOSS_MEAN, SWAY_LOSS_STD, SWAY_LOSS_LOWER, SWAY_LOSS_UPPER
    )[0]
    loss_factor = 1.0 / max(1e-6, (1.0 - sway_loss))

    # SES: 故障停机导致的产能折损
    if base_time_ses > 0:
        failures = rng.poisson(ELEVATOR_FAILURE_RATE * base_time_ses, size=N_ELEVATORS)
        downtime_days = failures * ELEVATOR_DOWNTIME_DAYS
        uptime_fractions = 1.0 - (downtime_days / (base_time_ses * 365.0))
        uptime_fractions = np.clip(uptime_fractions, 0.0, 1.0)
        capacity_factor = max(1e-6, uptime_fractions.mean())
    else:
        capacity_factor = 1.0

    time_ses = base_time_ses * loss_factor / capacity_factor
    cost_ses = (COST_SES_FIX * time_ses) + (COST_SES_VAR * mass_ses)

    # CRL: 发射次数与时间
    if mass_crl <= 0:
        time_crl = 0.0
        total_launches = 0
        cost_crl = 0.0
    else:
        required_success = int(np.ceil(mass_crl / LOAD_ROCKET))
        failures = rng.negative_binomial(required_success, 1.0 - ROCKET_FAIL_RATE)
        total_launches = required_success + failures
        flights_per_year = N_PADS * (365.0 / TURN_AROUND) * ROCKET_LAUNCH_PROB
        time_crl = total_launches / flights_per_year
        cost_crl = calc_learning_curve_cost(total_launches)

    time_total = max(time_ses, time_crl)
    cost_total = cost_ses + cost_crl

    return {
        "Alpha": alpha,
        "Time_Total": time_total,
        "Cost_Total": cost_total,
        "Time_SES": time_ses,
        "Time_CRL": time_crl,
        "Cost_SES": cost_ses,
        "Cost_CRL": cost_crl,
        "N_Launches": total_launches,
        "Sway_Loss": sway_loss,
        "Capacity_Factor": capacity_factor,
    }


def run_monte_carlo(alpha_values, n_sim=N_SIM, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    all_records = []
    for alpha in alpha_values:
        for sim in range(n_sim):
            metrics = simulate_system_metrics(alpha, rng)
            metrics["Sim"] = sim
            all_records.append(metrics)
    return pd.DataFrame(all_records)


def summarize_results(df, ci_level=0.95):
    z_score = 1.96 if np.isclose(ci_level, 0.95) else 1.645
    summary = (
        df.groupby("Alpha")
        .agg(
            Time_Mean=("Time_Total", "mean"),
            Time_Std=("Time_Total", "std"),
            Cost_Mean=("Cost_Total", "mean"),
            Cost_Std=("Cost_Total", "std"),
            Launches_Mean=("N_Launches", "mean"),
            Samples=("Time_Total", "count"),
        )
        .reset_index()
    )
    summary["Time_CI_Low"] = summary["Time_Mean"] - z_score * (
        summary["Time_Std"] / np.sqrt(summary["Samples"])
    )
    summary["Time_CI_High"] = summary["Time_Mean"] + z_score * (
        summary["Time_Std"] / np.sqrt(summary["Samples"])
    )
    summary["Cost_CI_Low"] = summary["Cost_Mean"] - z_score * (
        summary["Cost_Std"] / np.sqrt(summary["Samples"])
    )
    summary["Cost_CI_High"] = summary["Cost_Mean"] + z_score * (
        summary["Cost_Std"] / np.sqrt(summary["Samples"])
    )
    return summary


def plot_summary(summary_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(summary_df["Alpha"], summary_df["Time_Mean"], label="Mean Time")
    axes[0].fill_between(
        summary_df["Alpha"],
        summary_df["Time_CI_Low"],
        summary_df["Time_CI_High"],
        alpha=0.2,
        label="95% CI",
    )
    axes[0].set_xlabel("Alpha")
    axes[0].set_ylabel("Time (Years)")
    axes[0].set_title("Expected Time vs Alpha")
    axes[0].legend()

    axes[1].plot(summary_df["Alpha"], summary_df["Cost_Mean"], label="Mean Cost")
    axes[1].fill_between(
        summary_df["Alpha"],
        summary_df["Cost_CI_Low"],
        summary_df["Cost_CI_High"],
        alpha=0.2,
        label="95% CI",
    )
    axes[1].set_xlabel("Alpha")
    axes[1].set_ylabel("Cost (USD)")
    axes[1].set_title("Expected Cost vs Alpha")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig("Problem2_MonteCarlo_Summary.png", dpi=300)
    print("图表已保存至: Problem2_MonteCarlo_Summary.png")


def normalize_metrics(df_summary):
    raw_time = df_summary["Time_Mean"].values.reshape(-1, 1)
    raw_cost = df_summary["Cost_Mean"].values.reshape(-1, 1)

    time_mean = raw_time.mean()
    time_std = raw_time.std(ddof=0)
    cost_mean = raw_cost.mean()
    cost_std = raw_cost.std(ddof=0)

    norm_time = (raw_time - time_mean) / time_std
    norm_cost = (raw_cost - cost_mean) / cost_std

    return norm_time.flatten(), norm_cost.flatten()


def find_optimal_alpha(summary_df, gamma=GAMMA_DEFAULT):
    norm_time, norm_cost = normalize_metrics(summary_df)
    summary_df = summary_df.copy()
    summary_df["J_Score"] = gamma * norm_cost + (1.0 - gamma) * norm_time
    idx_opt = summary_df["J_Score"].idxmin()
    return summary_df, summary_df.loc[idx_opt]


def model2():
    alpha_values = np.linspace(0, 1, 101)
    raw_df = run_monte_carlo(alpha_values)
    summary_df = summarize_results(raw_df)

    summary_df, opt_sol = find_optimal_alpha(summary_df, GAMMA_DEFAULT)

    alpha_0 = summary_df.loc[summary_df["Alpha"] == 0.0].iloc[0]
    alpha_1 = summary_df.loc[summary_df["Alpha"] == 1.0].iloc[0]

    print("\n=== Monte Carlo Summary (Alpha=0) ===")
    print(alpha_0)
    print("\n=== Monte Carlo Summary (Alpha=1) ===")
    print(alpha_1)
    print("\n=== Optimal Alpha* under Uncertainty ===")
    print(opt_sol)

    plot_summary(summary_df)

    raw_df.to_csv("Problem2_MonteCarlo_Raw.csv", index=False)
    summary_df.to_csv("Problem2_MonteCarlo_Summary.csv", index=False)
    print("数据已保存至: Problem2_MonteCarlo_Raw.csv")
    print("数据已保存至: Problem2_MonteCarlo_Summary.csv")


if __name__ == "__main__":
    model2()
