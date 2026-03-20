import os
# 必须放在所有其他 import 之前！限制底层线程数，防止闪退/过度占用
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

import pandas as pd
import numpy as np
import math
import random
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity

# ==================================================
# Step 1. 加载数据与基础类型对齐
# ==================================================
print("==================================================")
print("Step 1. 正在加载数据...")
print("==================================================")

# 1.1 读取特征矩阵，指定第一列为索引
X_true = pd.read_csv('X_true.csv', index_col=0)
X_fixed = pd.read_csv('X_fixed.csv', index_col=0)
X_adaptive = pd.read_csv('X_adaptive.csv', index_col=0)

# 1.2 读取交互数据
train_df = pd.read_csv('Strict_Train_Interactions.csv')
val_df = pd.read_csv('val_df_processed.csv')
test_df = pd.read_csv('test_df_processed.csv')

# 1.3 类型统一，防止 mapping 错位
for df in [train_df, val_df, test_df]:
    df['reviewerID'] = df['reviewerID'].astype(str)
    df['asin'] = df['asin'].astype(str)

for matrix in [X_true, X_fixed, X_adaptive]:
    matrix.index = matrix.index.astype(str)

# 1.4 字段检查
required_cols = {'reviewerID', 'asin', 'Level_2'}
if not required_cols.issubset(set(train_df.columns)):
    raise ValueError(f"train_df 缺少必要字段: {required_cols - set(train_df.columns)}")

print(f"-> 矩阵加载成功! X_true: {X_true.shape}, X_fixed: {X_fixed.shape}, X_adaptive: {X_adaptive.shape}")
print(f"-> Train: {len(train_df)} 条 | Val: {len(val_df)} 条 | Test: {len(test_df)} 条")
print(f"-> Train 独立用户数: {train_df['reviewerID'].nunique()} | 独立商品数: {train_df['asin'].nunique()}")

# ==================================================
# Step 2. 预生成标准的 1+99 Sampled Candidate Set
# ==================================================
print("\n==================================================")
print("Step 2. 正在预生成标准的 1+99 Sampled Candidate Set...")
print("==================================================")

# 全局商品全集（仅来自 train）
all_train_items = set(train_df['asin'].unique())

# 每个用户在 train 中交互过的商品
user_seen_items_global = train_df.groupby('reviewerID')['asin'].apply(set).to_dict()

# item -> Level_2 映射（Soft Masking 必须用到）
item_to_category = (
    train_df.drop_duplicates(subset=['asin'])
    .set_index('asin')['Level_2']
    .to_dict()
)
print(f"-> 构建 item_to_category 完成，共 {len(item_to_category)} 个商品映射")

def generate_sampled_candidates(eval_df, name, num_negatives=99, seed=42):
    """
    为每个用户生成 1 个正样本 + 99 个负样本 的 sampled candidate set
    """
    random.seed(seed)
    sampled_dict = {}
    short_sample_count = 0

    for _, row in eval_df.iterrows():
        user = row['reviewerID']
        target_item = row['asin']

        seen_items = user_seen_items_global.get(user, set())
        valid_negatives_pool = list(all_train_items - seen_items - {target_item})

        if len(valid_negatives_pool) >= num_negatives:
            sampled_negatives = random.sample(valid_negatives_pool, num_negatives)
        else:
            sampled_negatives = valid_negatives_pool
            short_sample_count += 1

        candidate_set = set(sampled_negatives + [target_item])
        sampled_dict[user] = candidate_set

    print(f"-> [{name}] 1+{num_negatives} 候选集生成完毕! 共 {len(sampled_dict)} 个用户。")
    if short_sample_count > 0:
        print(f"    警告: 有 {short_sample_count} 个用户可用负样本不足 {num_negatives} 个。")

    return sampled_dict

val_sampled_candidates = generate_sampled_candidates(val_df, "Validation Set")
test_sampled_candidates = generate_sampled_candidates(test_df, "Test Set")

# ==================================================
# Step 3. 工具函数：构建内存安全版 base clusters
# ==================================================
def build_base_clusters(X_matrix):
    """
    构建多组 base clustering：
    - KMeans: k=6,8,10
    - DBSCAN: eps=0.3,0.4,0.5
    返回：
    - base_clusters: list[set of member indices]
    - base_centroids: np.array
    """
    X = X_matrix.values
    n_users = len(X)

    base_clusters = []
    base_centroids = []

    print("  -> Base Clustering: KMeans (k=6,8,10)")
    for k in [6, 8, 10]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            base_clusters.append(set(members))
            base_centroids.append(X[members].mean(axis=0))

    print("  -> Base Clustering: DBSCAN (eps=0.3,0.4,0.5)")
    min_samples = max(2, int(0.001 * n_users))
    for eps in [0.3, 0.4, 0.5]:
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = db.fit_predict(X)

        for c in set(labels):
            if c == -1:
                continue
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            base_clusters.append(set(members))
            base_centroids.append(X[members].mean(axis=0))

    return base_clusters, np.array(base_centroids, dtype=np.float32)

# ==================================================
# Step 4. 工具函数：构建 base-cluster 相似度矩阵 W
# 内存安全：不再构造用户两两相似度矩阵
# ==================================================
def build_super_cluster_similarity(base_clusters, base_centroids):
    """
    用 base cluster 质心相似度 + 成员重叠度 构造 W
    这是内存安全版的近似 LDPEC 风格实现
    """
    num_base = len(base_clusters)
    W = np.zeros((num_base, num_base), dtype=np.float32)

    # 质心余弦相似度
    centroid_sim = cosine_similarity(base_centroids)

    # 衰减因子 beta，控制大簇影响
    betas = np.array([1.0 / (1.0 + np.log(len(c))) for c in base_clusters], dtype=np.float32)

    for i in range(num_base):
        set_i = base_clusters[i]
        len_i = len(set_i)

        for j in range(i, num_base):
            set_j = base_clusters[j]
            len_j = len(set_j)

            # 成员 Jaccard 重叠
            inter = len(set_i & set_j)
            union = len_i + len_j - inter
            overlap = inter / union if union > 0 else 0.0

            # 组合相似度
            sim_val = 0.7 * max(float(centroid_sim[i, j]), 0.0) + 0.3 * overlap
            sim_val = betas[i] * betas[j] * sim_val

            W[i, j] = sim_val
            W[j, i] = sim_val

    return W

# ==================================================
# Step 5. 工具函数：用户重分配到 Final Cluster
# ==================================================
def assign_users_to_final_clusters(X_matrix, base_clusters, super_labels, K_clusters):
    """
    先将 base clusters 合并成 final clusters
    再根据用户与 final cluster centroid 的余弦相似度分配用户
    """
    X = X_matrix.values

    final_cluster_members = {k: set() for k in range(K_clusters)}
    for base_idx, final_idx in enumerate(super_labels):
        final_cluster_members[final_idx].update(base_clusters[base_idx])

    # 构建 final cluster centroids
    final_centroids = []
    for k in range(K_clusters):
        members = list(final_cluster_members[k])
        if len(members) == 0:
            final_centroids.append(np.zeros(X.shape[1], dtype=np.float32))
        else:
            final_centroids.append(X[members].mean(axis=0))

    final_centroids = np.array(final_centroids, dtype=np.float32)

    # 用户到 final centroid 的相似度
    user_cluster_sim = cosine_similarity(X, final_centroids)
    final_user_labels = np.argmax(user_cluster_sim, axis=1)

    return final_user_labels

# ==================================================
# Step 6. 定义最终 Ensemble + Soft Masking + Direct Scoring 流水线
# ==================================================
def run_ensemble_profile_mask_sampled_direct(
    X_matrix,
    model_name,
    train_df,
    eval_df,
    sampled_candidates,
    K_clusters=50,
    Top_K_Eval=10,
    lambda_penalty=0.2
):
    print(f"\n[{model_name}]  开始执行 Ensemble + Soft Masking 评估 (Lambda={lambda_penalty})...")

    # ------------------------------
    # 1. Base Clustering
    # ------------------------------
    print("  -> 1. 执行 Base Clustering...")
    base_clusters, base_centroids = build_base_clusters(X_matrix)
    print(f"     共生成 {len(base_clusters)} 个 base clusters")

    # ------------------------------
    # 2. 构建 super-cluster 相似度矩阵 W
    # ------------------------------
    print("  -> 2. 构建 Base Cluster 相似度矩阵 W...")
    W = build_super_cluster_similarity(base_clusters, base_centroids)

    # ------------------------------
    # 3. Spectral Super-Clustering
    # ------------------------------
    print("  -> 3. 执行 SpectralClustering (Super-Clustering)...")
    sc = SpectralClustering(
        n_clusters=K_clusters,
        affinity='precomputed',
        random_state=42,
        assign_labels='kmeans'
    )
    super_labels = sc.fit_predict(W)

    # ------------------------------
    # 4. 用户重分配到 Final Cluster
    # ------------------------------
    print("  -> 4. 用户重分配到 Final Cluster...")
    final_user_labels = assign_users_to_final_clusters(X_matrix, base_clusters, super_labels, K_clusters)
    user_cluster_map = dict(zip(X_matrix.index, final_user_labels))

    # ------------------------------
    # 5. 统计 Cluster 内部商品热度
    # ------------------------------
    print("  -> 5. 统计 Cluster 内部商品热度...")
    train_cluster_df = train_df[train_df['reviewerID'].isin(user_cluster_map.keys())].copy()
    train_cluster_df['Cluster_ID'] = train_cluster_df['reviewerID'].map(user_cluster_map)

    cluster_item_pop = (
        train_cluster_df.groupby(['Cluster_ID', 'asin'])
        .size()
        .reset_index(name='pop_count')
    )

    cluster_pop_dict = {}
    for c_id in range(K_clusters):
        c_items = cluster_item_pop[cluster_item_pop['Cluster_ID'] == c_id]
        cluster_pop_dict[c_id] = dict(zip(c_items['asin'], c_items['pop_count']))

    # ------------------------------
    # 6. 对 Sampled Candidates 全量打分 (Soft Masking)
    # ------------------------------
    print("  -> 6. 正在对 sampled candidates 直接打分并排序 (Soft Masking)...")
    hits = 0
    ndcg_sum = 0.0
    valid_users = 0
    missing_cluster_users = 0
    target_not_in_sampled = 0
    uncovered_targets = 0

    for _, row in eval_df.iterrows():
        user = row['reviewerID']
        target_item = row['asin']

        c_id = user_cluster_map.get(user)
        if c_id is None:
            missing_cluster_users += 1
            continue

        sampled_list = sampled_candidates.get(user, None)
        if not sampled_list:
            continue

        if target_item not in sampled_list:
            target_not_in_sampled += 1
            continue

        valid_users += 1

        seen_items = user_seen_items_global.get(user, set())
        user_profile = X_matrix.loc[user].to_dict()
        candidate_pop_dict = cluster_pop_dict.get(c_id, {})

        scored_items = []
        for item in sampled_list:
            # 非 target 且已见则过滤
            if item != target_item and item in seen_items:
                continue

            item_cat = item_to_category.get(item, None)
            pop_score = candidate_pop_dict.get(item, 0)

            # Soft Masking
            if item_cat is None:
                mask_weight = lambda_penalty
            else:
                x_val = user_profile.get(item_cat, 0)  # 当前仍为 0/1
                mask_weight = lambda_penalty + (1.0 - lambda_penalty) * x_val

            final_score = pop_score * mask_weight
            scored_items.append((item, final_score))

        # 排序：分数相同则按 item id 次序打破平局
        scored_items.sort(key=lambda x: (-x[1], x[0]))
        ranked_items = [x[0] for x in scored_items]
        top_k_items = ranked_items[:Top_K_Eval]

        if target_item in top_k_items:
            hits += 1
            rank = top_k_items.index(target_item)
            dcg = 1.0 / math.log2(rank + 2)
            ndcg_sum += dcg
        else:
            uncovered_targets += 1

    hr_score = hits / valid_users if valid_users > 0 else 0.0
    ndcg_score = ndcg_sum / valid_users if valid_users > 0 else 0.0

    print(f"   -> Valid Users: {valid_users} | Hits@{Top_K_Eval}: {hits}")
    if missing_cluster_users > 0:
        print(f"   ->  有 {missing_cluster_users} 个用户缺失 Cluster 映射。")
    if target_not_in_sampled > 0:
        print(f"   ->  有 {target_not_in_sampled} 个用户的 target 不在 sampled set 中。")
    if uncovered_targets > 0:
        print(f"   ->  有 {uncovered_targets} 个用户的真实商品未进入 Top-{Top_K_Eval}。")

    print(f"    Result | Sampled HR@{Top_K_Eval}: {hr_score:.4f} | Sampled NDCG@{Top_K_Eval}: {ndcg_score:.4f}")

    return hr_score, ndcg_score

# ==================================================
# Step 7. 分别在 Val 和 Test 上运行三种模型 (搭配专属 Lambda)
# ==================================================
TOP_K = 10
K_CLUSTERS = 8  # 最终簇数量保持为 8

models_config = [
    ("No-Noise (Upper Bound)", X_true, 0.2),
    ("Fixed LDP (Baseline)", X_fixed, 0.2),
    ("Adaptive LDP (Ours)", X_adaptive, 0.3)
]

results_val = []
results_test = []

print("\n==================================================")
print("Step 7a. 开始 Validation Set (Ensemble + Soft Masking) 评估")
print("==================================================")

for model_name, X, current_lambda in models_config:
    hr, ndcg = run_ensemble_profile_mask_sampled_direct(
        X_matrix=X,
        model_name=model_name,
        train_df=train_df,
        eval_df=val_df,
        sampled_candidates=val_sampled_candidates,
        K_clusters=K_CLUSTERS,
        Top_K_Eval=TOP_K,
        lambda_penalty=current_lambda
    )
    results_val.append({
        "Model": model_name,
        "Lambda": current_lambda,
        f"HR@{TOP_K}": hr,
        f"NDCG@{TOP_K}": ndcg
    })

print("\n==================================================")
print("Step 7b. 开始 Test Set (Ensemble + Soft Masking) 评估")
print("==================================================")

for model_name, X, current_lambda in models_config:
    hr, ndcg = run_ensemble_profile_mask_sampled_direct(
        X_matrix=X,
        model_name=model_name,
        train_df=train_df,
        eval_df=test_df,
        sampled_candidates=test_sampled_candidates,
        K_clusters=K_CLUSTERS,
        Top_K_Eval=TOP_K,
        lambda_penalty=current_lambda
    )
    results_test.append({
        "Model": model_name,
        "Lambda": current_lambda,
        f"HR@{TOP_K}": hr,
        f"NDCG@{TOP_K}": ndcg
    })

# ==================================================
# Step 8. 打印最终大表
# ==================================================
df_res_val = pd.DataFrame(results_val)
df_res_test = pd.DataFrame(results_test)

print("\n" + "=" * 70)
print(" VALIDATION PERFORMANCE (LDPEC-style Ensemble + Soft Masking)")
print("=" * 70)
print(df_res_val.to_string(index=False))

print("\n" + "=" * 70)
print(" TEST PERFORMANCE (LDPEC-style Ensemble + Soft Masking)")
print("=" * 70)
print(df_res_test.to_string(index=False))
print("=" * 70)