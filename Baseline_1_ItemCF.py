import os
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '8'

import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==================================================
# Config
# ==================================================
TOP_K = 10
NUM_NEGATIVES = 99
SEED = 42
SIM_TYPE = 'cosine'   # 'cosine', 'jaccard', 'phi'
EPS_FIXED = 1.6725    # align with your fixed-LDP setting

TRAIN_PATH = 'Strict_Train_Interactions.csv'
VAL_PATH = 'val_df_processed.csv'
TEST_PATH = 'test_df_processed.csv'
TRUE_PATH = 'X_true.csv'
FIXED_PATH = 'X_fixed.csv'
ADAPTIVE_PATH = 'X_adaptive.csv'

# ==================================================
# Step 1. Load data and align types
# ==================================================
print('==================================================')
print('Step 1. Loading data...')
print('==================================================')

X_true = pd.read_csv(TRUE_PATH, index_col=0)
X_fixed = pd.read_csv(FIXED_PATH, index_col=0)
X_adaptive = pd.read_csv(ADAPTIVE_PATH, index_col=0)

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

for df in [train_df, val_df, test_df]:
    df['reviewerID'] = df['reviewerID'].astype(str)
    df['asin'] = df['asin'].astype(str)
    df['Level_2'] = df['Level_2'].astype(str)

for matrix in [X_true, X_fixed, X_adaptive]:
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)  # Level_2 labels

required_cols = {'reviewerID', 'asin', 'Level_2'}
for name, df in [('train_df', train_df), ('val_df', val_df), ('test_df', test_df)]:
    miss = required_cols - set(df.columns)
    if miss:
        raise ValueError(f'{name} missing required columns: {miss}')

print(f'-> X_true: {X_true.shape}, X_fixed: {X_fixed.shape}, X_adaptive: {X_adaptive.shape}')
print(f'-> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}')
print(f'-> Train users: {train_df["reviewerID"].nunique()} | Train items: {train_df["asin"].nunique()}')

# ==================================================
# Step 2. Build standard 1+99 sampled candidate sets
# ==================================================
print('\n==================================================')
print('Step 2. Building standard 1+99 sampled candidate sets...')
print('==================================================')

all_train_items = set(train_df['asin'].unique())
user_seen_items_global = train_df.groupby('reviewerID')['asin'].apply(set).to_dict()

# item -> category mapping, for turning category scores into item scores
item_to_category = (
    train_df.drop_duplicates(subset=['asin'])
    .set_index('asin')['Level_2']
    .astype(str)
    .to_dict()
)

print(f'-> Built item_to_category for {len(item_to_category)} items')

def generate_sampled_candidates(eval_df, name, num_negatives=99, seed=42):
    random.seed(seed)
    sampled_dict = {}
    short_sample_count = 0

    for _, row in eval_df.iterrows():
        user = str(row['reviewerID'])
        target_item = str(row['asin'])
        seen_items = user_seen_items_global.get(user, set())
        valid_neg_pool = list(all_train_items - seen_items - {target_item})

        if len(valid_neg_pool) >= num_negatives:
            sampled_negatives = random.sample(valid_neg_pool, num_negatives)
        else:
            sampled_negatives = valid_neg_pool
            short_sample_count += 1

        sampled_dict[user] = set(sampled_negatives + [target_item])

    print(f'-> [{name}] generated for {len(sampled_dict)} users')
    if short_sample_count > 0:
        print(f'   warning: {short_sample_count} users have < {num_negatives} negatives available')
    return sampled_dict

val_sampled_candidates = generate_sampled_candidates(val_df, 'Validation', NUM_NEGATIVES, SEED)
test_sampled_candidates = generate_sampled_candidates(test_df, 'Test', NUM_NEGATIVES, SEED)

# 【新增】：在函数外部（全局）计算每个 Item 的流行度
# 这个可以放在 Step 2 之后
item_popularity = train_df['asin'].value_counts().to_dict()
max_pop = max(item_popularity.values()) if item_popularity else 1.0


def evaluate_sampled_itemcf(eval_df, sampled_candidates, user_seen_categories, item_to_category, cat_sim, top_k=10):
    hits = 0
    ndcg_sum = 0.0
    valid_users = 0
    target_not_in_sampled = 0

    cat_sim_dict = cat_sim.to_dict()

    for row in tqdm(eval_df.itertuples(), total=len(eval_df)):
        user = str(row.reviewerID)
        target_item = str(row.asin)

        candidate_set = sampled_candidates.get(user, None)
        if not candidate_set:
            continue
        if target_item not in candidate_set:
            target_not_in_sampled += 1
            continue

        candidate_items = list(candidate_set)
        hist_cats = list(user_seen_categories.get(user, set()))

        scores = np.zeros(len(candidate_items), dtype=np.float64)
        
        for idx, item in enumerate(candidate_items):
            item_cat = item_to_category.get(item)
            s = 0.0
            
            # 1. 计算类别相似度基础分 (Category-level score)
            if hist_cats and item_cat:
                for hc in hist_cats:
                    s += cat_sim_dict.get(hc, {}).get(item_cat, 0.0)
            
            # 2. 【核心修正】：Item Popularity Tie-breaker
            # 除以 max_pop 归一化，乘 1e-6 保证它极小，只在 s 平分时起作用
            pop_bonus = (item_popularity.get(item, 0) / max_pop) * 1e-6
            
            scores[idx] = s + pop_bonus

        # 排序取 Top-K
        ranked_idx = np.argsort(-scores)
        ranked_items = np.array(candidate_items)[ranked_idx]
        top_items = ranked_items[:top_k]

        valid_users += 1
        if target_item in top_items:
            hits += 1
            rank = np.where(top_items == target_item)[0][0]
            ndcg_sum += 1.0 / math.log2(rank + 2)

    hr = hits / valid_users if valid_users > 0 else 0.0
    ndcg = ndcg_sum / valid_users if valid_users > 0 else 0.0
    return hr, ndcg, valid_users, target_not_in_sampled

# ==================================================
# Step 3. LDP-ItemCF: unbiased reconstruction at category level
# ==================================================
def debias_binary_matrix(Y, epsilon):
    """
    Bit-wise randomized response unbiased estimator:
        x_hat = (y - q) / (p - q)
    where p = e^eps / (1+e^eps), q = 1 / (1+e^eps)
    """
    p = np.exp(epsilon) / (1.0 + np.exp(epsilon))
    q = 1.0 / (1.0 + np.exp(epsilon))
    denom = p - q
    if abs(denom) < 1e-12:
        raise ValueError('epsilon too small; p-q is numerically unstable')
    X_hat = (Y - q) / denom
    return X_hat.astype(np.float32), p, q


def reconstruct_item_statistics(X_hat, clip=True):
    """
    Recover per-category marginal counts and pairwise co-occurrence counts.
    n1[c]    = sum_u xhat_{u,c}
    n11[c,d] = sum_u xhat_{u,c} * xhat_{u,d}
    """
    n_users = X_hat.shape[0]
    n1 = X_hat.sum(axis=0)
    n11 = X_hat.T @ X_hat
    if clip:
        n1 = np.clip(n1, 0.0, float(n_users))
        n11 = np.clip(n11, 0.0, float(n_users))
    return n1.astype(np.float32), n11.astype(np.float32)


def reconstruct_pairwise_4counts(n1, n11, num_users, clip=True):
    n10 = n1[:, None] - n11
    n01 = n1[None, :] - n11
    n00 = float(num_users) - n11 - n10 - n01
    if clip:
        n10 = np.clip(n10, 0.0, float(num_users))
        n01 = np.clip(n01, 0.0, float(num_users))
        n00 = np.clip(n00, 0.0, float(num_users))
    return (
        n11.astype(np.float32),
        n10.astype(np.float32),
        n01.astype(np.float32),
        n00.astype(np.float32),
    )


def compute_item_similarity(n11, n10, n01, n00, sim_type='cosine'):
    eps = 1e-12
    if sim_type == 'cosine':
        denom = np.sqrt((n11 + n10) * (n11 + n01)) + eps
        sim = n11 / denom
    elif sim_type == 'jaccard':
        denom = (n11 + n10 + n01) + eps
        sim = n11 / denom
    elif sim_type == 'phi':
        numerator = n11 * n00 - n10 * n01
        denom = np.sqrt((n11 + n10) * (n11 + n01) * (n00 + n10) * (n00 + n01)) + eps
        sim = numerator / denom
    else:
        raise ValueError(f'Unsupported sim_type: {sim_type}')

    np.fill_diagonal(sim, 1.0)
    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
    return sim.astype(np.float32)


def build_user_seen_categories(train_df):
    return train_df.groupby('reviewerID')['Level_2'].apply(lambda x: set(map(str, x))).to_dict()


def score_candidates_by_category_itemcf(user, candidate_items, user_seen_categories, item_to_category, cat_sim):
    """
    Turn category-level ItemCF into item ranking scores.
    score(u, item) = sum_{c in HistCat(u)} sim(cat(item), c)
    """
    hist_cats = list(user_seen_categories.get(user, set()))
    if len(hist_cats) == 0:
        return np.zeros(len(candidate_items), dtype=np.float32)

    scores = np.zeros(len(candidate_items), dtype=np.float32)
    for idx, item in enumerate(candidate_items):
        item_cat = item_to_category.get(item, None)
        if item_cat is None:
            scores[idx] = 0.0
            continue
        if item_cat not in cat_sim.index:
            scores[idx] = 0.0
            continue
        scores[idx] = float(cat_sim.loc[item_cat, hist_cats].sum())
    return scores


def evaluate_sampled_itemcf(eval_df, sampled_candidates, user_seen_categories, item_to_category, cat_sim, top_k=10):
    hits = 0
    ndcg_sum = 0.0
    valid_users = 0
    target_not_in_sampled = 0

    # 【提速核心 1】：将 Pandas DataFrame 预先转化为原生 Python 嵌套字典
    # 结构为: dict[列名][行名] -> 查表时间复杂度降至 O(1)
    cat_sim_dict = cat_sim.to_dict()

    # 【提速核心 2】：废弃 iterrows，使用 itertuples 加速外层遍历
    for row in tqdm(eval_df.itertuples(), total=len(eval_df)):
        user = str(row.reviewerID)
        target_item = str(row.asin)

        candidate_set = sampled_candidates.get(user, None)
        if not candidate_set:
            continue
        if target_item not in candidate_set:
            target_not_in_sampled += 1
            continue

        candidate_items = list(candidate_set)
        hist_cats = list(user_seen_categories.get(user, set()))

        # 【提速核心 3】：完全在 Numpy 和 Dict 层面进行内层打分
        scores = np.zeros(len(candidate_items), dtype=np.float32)
        if hist_cats:
            for idx, item in enumerate(candidate_items):
                item_cat = item_to_category.get(item)
                if item_cat:
                    s = 0.0
                    for hc in hist_cats:
                        # 极速查表：cat_sim_dict[hc] 获取的是列，item_cat 是行
                        s += cat_sim_dict.get(hc, {}).get(item_cat, 0.0)
                    scores[idx] = s

        # 排序取 Top-K
        ranked_idx = np.argsort(-scores)
        ranked_items = np.array(candidate_items)[ranked_idx]
        top_items = ranked_items[:top_k]

        valid_users += 1
        if target_item in top_items:
            hits += 1
            rank = np.where(top_items == target_item)[0][0]
            ndcg_sum += 1.0 / math.log2(rank + 2)

    hr = hits / valid_users if valid_users > 0 else 0.0
    ndcg = ndcg_sum / valid_users if valid_users > 0 else 0.0
    return hr, ndcg, valid_users, target_not_in_sampled


def run_ldp_itemcf_category_pipeline(X_matrix, model_name, train_df, eval_df, sampled_candidates,
                                     epsilon=EPS_FIXED, sim_type=SIM_TYPE, top_k=TOP_K):
    print(f'\n[{model_name}] Running LDP-ItemCF (unbiased reconstruction) ...')

    # Align users and categories to X_matrix
    user_index = X_matrix.index.astype(str)
    category_cols = X_matrix.columns.astype(str)
    Y = X_matrix.loc[user_index, category_cols].values.astype(np.float32)

    if 'No-Noise' in model_name:
        X_hat = Y  # 真实矩阵无需去噪
        print('  -> Using raw matrix without debiasing.')
    else:
        X_hat, p, q = debias_binary_matrix(Y, epsilon=epsilon)
        print(f'  -> Debiased with epsilon={epsilon:.4f} | p={p:.6f} | q={q:.6f}')

    n1, n11 = reconstruct_item_statistics(X_hat, clip=True)
    n11, n10, n01, n00 = reconstruct_pairwise_4counts(n1, n11, num_users=Y.shape[0], clip=True)
    sim = compute_item_similarity(n11, n10, n01, n00, sim_type=sim_type)

    cat_sim = pd.DataFrame(sim, index=category_cols, columns=category_cols)
    user_seen_categories = build_user_seen_categories(train_df)

    hr, ndcg, valid_users, missed = evaluate_sampled_itemcf(
        eval_df=eval_df,
        sampled_candidates=sampled_candidates,
        user_seen_categories=user_seen_categories,
        item_to_category=item_to_category,
        cat_sim=cat_sim,
        top_k=top_k,
    )

    print(f'   valid_users={valid_users} | target_not_in_sampled={missed}')
    print(f'   Result | HR@{top_k}: {hr:.4f} | NDCG@{top_k}: {ndcg:.4f}')
    return hr, ndcg

# ==================================================
# Step 4. Run baselines
# ==================================================
models_config = [
    ('No-Noise Category-ItemCF', X_true, EPS_FIXED),
    ('Fixed LDP-ItemCF', X_fixed, EPS_FIXED),
    # note: adaptive matrix was generated with per-entry eps, but exact unbiased
    # reconstruction for mixed eps requires entry-wise correction. As a clean baseline,
    # we evaluate the paper-faithful fixed-epsilon LDP-ItemCF here.
]

results_val = []
results_test = []

print('\n==================================================')
print('Step 4a. Validation evaluation')
print('==================================================')
for model_name, X, eps in models_config:
    hr, ndcg = run_ldp_itemcf_category_pipeline(
        X_matrix=X,
        model_name=model_name,
        train_df=train_df,
        eval_df=val_df,
        sampled_candidates=val_sampled_candidates,
        epsilon=eps,
        sim_type=SIM_TYPE,
        top_k=TOP_K,
    )
    results_val.append({
        'Model': model_name,
        'HR@10': hr,
        'NDCG@10': ndcg,
    })

print('\n==================================================')
print('Step 4b. Test evaluation')
print('==================================================')
for model_name, X, eps in models_config:
    hr, ndcg = run_ldp_itemcf_category_pipeline(
        X_matrix=X,
        model_name=model_name,
        train_df=train_df,
        eval_df=test_df,
        sampled_candidates=test_sampled_candidates,
        epsilon=eps,
        sim_type=SIM_TYPE,
        top_k=TOP_K,
    )
    results_test.append({
        'Model': model_name,
        'HR@10': hr,
        'NDCG@10': ndcg,
    })

res_val = pd.DataFrame(results_val)
res_test = pd.DataFrame(results_test)

print('\n' + '=' * 70)
print(' VALIDATION PERFORMANCE (LDP-ItemCF with Unbiased Reconstruction)')
print('=' * 70)
print(res_val.to_string(index=False))

print('\n' + '=' * 70)
print(' TEST PERFORMANCE (LDP-ItemCF with Unbiased Reconstruction)')
print('=' * 70)
print(res_test.to_string(index=False))
print('=' * 70)

res_val.to_csv('ldp_itemcf_val_results.csv', index=False)
res_test.to_csv('ldp_itemcf_test_results.csv', index=False)
print('\nSaved: ldp_itemcf_val_results.csv, ldp_itemcf_test_results.csv')

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm
import math

# 配置 (请确保与之前一致)
TOP_K = 10
EPS_FIXED = 1.6725

# 1. 准备工作
print("Loading data...")
X_fixed = pd.read_csv('X_fixed.csv', index_col=0)
train_df = pd.read_csv('Strict_Train_Interactions.csv')
test_df = pd.read_csv('test_df_processed.csv') # 评测集

# 计算全局真实的 Item Popularity (用于 Tie-breaker 和 LDP-Pop 映射)
item_popularity = train_df['asin'].value_counts().to_dict()
max_pop = max(item_popularity.values()) if item_popularity else 1.0
item_to_category = train_df.drop_duplicates(subset=['asin']).set_index('asin')['Level_2'].astype(str).to_dict()

# 我们需要复用你之前的 1+99 生成函数，这里假设 test_sampled_candidates 已生成
# test_sampled_candidates = ... (此处省略，直接用你之前的 dict)

# ==========================================
# 新基线 1: LDP-Pop (无偏全局流行度推荐)
# ==========================================
print("\n[Baseline] Running LDP-Pop...")
p = np.exp(EPS_FIXED) / (1.0 + np.exp(EPS_FIXED))
q = 1.0 / (1.0 + np.exp(EPS_FIXED))
N_users = X_fixed.shape[0]

# 1. 重构 Category 级别的频数
noisy_cat_counts = X_fixed.sum(axis=0)
unbiased_cat_counts = (noisy_cat_counts - N_users * q) / (p - q)
unbiased_cat_counts = np.clip(unbiased_cat_counts, 0, N_users).to_dict()

# 2. 为每个 Target 打分 (只看全局热度)
hits_pop = 0
ndcg_pop = 0.0
valid_users_pop = 0

for row in tqdm(test_df.itertuples(), total=len(test_df)):
    user = str(row.reviewerID)
    target_item = str(row.asin)
    candidate_set = test_sampled_candidates.get(user)
    if not candidate_set or target_item not in candidate_set: continue
    
    candidate_items = list(candidate_set)
    scores = np.zeros(len(candidate_items), dtype=np.float64)
    
    for idx, item in enumerate(candidate_items):
        cat = item_to_category.get(item)
        # 类别重构热度 + 细粒度商品热度(Tie-breaker)
        cat_pop = unbiased_cat_counts.get(cat, 0.0) 
        item_bonus = (item_popularity.get(item, 0) / max_pop) * 1e-6
        scores[idx] = cat_pop + item_bonus
        
    ranked_items = np.array(candidate_items)[np.argsort(-scores)]
    top_items = ranked_items[:TOP_K]
    valid_users_pop += 1
    if target_item in top_items:
        hits_pop += 1
        rank = np.where(top_items == target_item)[0][0]
        ndcg_pop += 1.0 / math.log2(rank + 2)

print(f"-> LDP-Pop HR@10: {hits_pop/valid_users_pop:.4f} | NDCG@10: {ndcg_pop/valid_users_pop:.4f}")
