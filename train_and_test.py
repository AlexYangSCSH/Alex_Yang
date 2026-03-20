import pandas as pd
import json
import os
import numpy as np

def parse_amazon_metadata(file_path, max_rows=None):
    """
    安全读取 Amazon Metadata JSON 文件，并提取 ASIN 和前三级类目
    :param file_path: metadata 文件的本地路径
    :param max_rows: 限制读取的行数（用于本地代码测试，设为 None 则读取全量）
    """
    print(f"开始解析文件: {file_path} ...")
    
    extracted_data = []
    
    # 使用逐行读取，防止内存爆掉，也方便跳过损坏的行
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # 如果设置了最大行数限制，方便快速测试
            if max_rows and i >= max_rows:
                break
                
            try:
                # 解析单行 JSON
                data = json.loads(line.strip())
                
                # 提取核心字段：商品ID (asin) 和 类目树 (category)
                asin = data.get('asin', '')
                categories = data.get('category', [])
                
                # 很多商品可能没有类目信息，我们需要清洗掉这些“孤儿”数据
                if not asin or not isinstance(categories, list) or len(categories) == 0:
                    continue
                
                # 安全提取前三级类目（如果没有那么多级，则补 'None'）
                level_1 = categories[0] if len(categories) > 0 else 'None'
                level_2 = categories[1] if len(categories) > 1 else 'None'
                level_3 = categories[2] if len(categories) > 2 else 'None'
                
                # 剔除掉那些含有 HTML 标签的脏数据 (比如 </span>)
                if '</span>' in level_1 or '</span>' in level_2:
                    continue
                    
                extracted_data.append({
                    'asin': asin,
                    'Level_1': level_1,
                    'Level_2': level_2,
                    'Level_3': level_3
                })
                
            except json.JSONDecodeError:
                # 忽略极个别格式损坏的行
                continue
                
    # 将提取出的干净数据转化为 Pandas DataFrame
    meta_df = pd.DataFrame(extracted_data)
    print(f"解析完成！共提取了 {len(meta_df)} 个有效商品的类目信息。")
    return meta_df

# ==========================================
# 🚀 使用方法 (你只需要修改下面的文件路径)
# ==========================================

# 假设你下载的 metadata 文件叫 'meta_Video_Games.json'
# 这里为了你能快速测试代码是否报错，我设置了 max_rows=50000
# 等你正式跑实验时，把 max_rows 改成 None 即可全量读取
FILE_PATH = 'meta_Video_Games.json' 

# 检查文件是否存在，如果不存在则提示
if os.path.exists(FILE_PATH):
    df_taxonomy = parse_amazon_metadata(FILE_PATH, max_rows=None)
    # 修复 HTML 转义字符
    df_taxonomy['Level_2'] = df_taxonomy['Level_2'].str.replace('&amp;', '&')
    df_taxonomy['Level_1'] = df_taxonomy['Level_1'].str.replace('&amp;', '&')
    # 打印前 10 行看看你的“商品类目字典”长什么样
    print("\n--- 你的 Taxonomy Tree (前 10 个商品) ---")
    print(df_taxonomy.head(10))
    
    # 统计一下 Level_2 (中类) 到底有多少个不同的维度
    print(f"\nLevel_2 一共有 {df_taxonomy['Level_2'].nunique()} 个独立类目。")
    print(f"最常见的 {df_taxonomy['Level_2'].nunique()} 个 Level_2 类目是：")
    print(df_taxonomy['Level_2'].value_counts().head(df_taxonomy['Level_2'].nunique()))
    
else:
    print(f"⚠️ 找不到文件 {FILE_PATH}！请确保你已经将下载好的 JSON 文件放在了代码同级目录下。")

REVIEWS_PATH = 'Video_Games_5.json'

print("1. 开始读取交互数据并提取时间戳与原始顺序...")
reviews_data = []
with open(REVIEWS_PATH, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f): # 【解决隐患 A】：加上 enumerate 提取原始行号 i
        try:
            data = json.loads(line.strip())
            reviews_data.append({
                'reviewerID': data.get('reviewerID'),
                'asin': data.get('asin'),
                'timestamp': data.get('unixReviewTime', 0),
                'orig_order': i  # 【解决隐患 A】：记录 JSON 文件中的原始读取顺序
            })
        except json.JSONDecodeError:
            continue

df_reviews = pd.DataFrame(reviews_data)
print(f"-> 读取完成！初始共提取 {len(df_reviews)} 条记录。")

# ==========================================
# 2. 【解决隐患 B】：先 Merge 类目，再过滤无效用户
# ==========================================
print("\n2. 正在合并类目树并过滤无效用户 (保证划分所需的最低交互数)...")

# 假设 df_taxonomy 已经在你的内存中 (就是你之前跑的 parse_amazon_metadata 的结果)
# 只保留有类目信息的交互记录
merged_df = pd.merge(df_reviews, df_taxonomy[['asin', 'Level_2']], on='asin', how='inner')

# 因为我们需要 Train(至少1条), Val(1条), Test(1条)，所以每个用户必须至少保留 3 条记录
user_counts = merged_df['reviewerID'].value_counts()
valid_users = user_counts[user_counts >= 3].index
merged_df = merged_df[merged_df['reviewerID'].isin(valid_users)].copy()

print(f"-> 清洗完成！保留了 {len(valid_users)} 个有效用户，共 {len(merged_df)} 条完整记录。")

# ==========================================
# 3. 【解决隐患 C】：严格排序与 Leave-Two-Out 划分
# ==========================================
print("\n3. 正在按用户时间线切分训练集(Train)、验证集(Val)与测试集(Test)...")

# 【解决隐患 A】：增加 orig_order 作为第二排序键，防止时间戳完全相同导致顺序随机跳动
merged_df = merged_df.sort_values(by=['reviewerID', 'timestamp', 'orig_order'])

# 取每个用户最后一次交互作为 Test
test_df = merged_df.groupby('reviewerID').tail(1)
remaining_df = merged_df.drop(test_df.index)

# 取每个用户倒数第二次交互作为 Validation (用于后续调参，防止过拟合)
val_df = remaining_df.groupby('reviewerID').tail(1)

# 剩下的所有交互作为 Train
train_df = remaining_df.drop(val_df.index)

print(f"-> 划分完成！")
print(f"   Train Set (训练集): {len(train_df)} 条记录 (仅用于特征提取和聚类)")
print(f"   Val Set   (验证集): {len(val_df)} 条记录 (等于总用户数, 用于调参)")
print(f"   Test Set  (测试集): {len(test_df)} 条记录 (等于总用户数, 用于最终评估)")

# ==========================================
# 4. 【解决隐患 D】：绝对隔离！仅基于 Train 计算流行度
# ==========================================
print("\n4. 正在仅基于 Train 集统计商品流行度 (防止未来数据泄露)...")
item_popularity = train_df['asin'].value_counts().reset_index()
item_popularity.columns = ['asin', 'interact_count']

# 使用 pd.qcut 按分位数切分：前 20% 为 Head，中间 30% 为 Torso，后 50% 为 Tail
item_popularity['Popularity_Group'] = pd.qcut(
    item_popularity['interact_count'].rank(method='first', ascending=False), 
    q=[0, 0.2, 0.5, 1.0], 
    labels=['Head', 'Torso', 'Tail']
)

# 把算好的流行度标签只贴给 Train 集
train_df = pd.merge(train_df, item_popularity[['asin', 'Popularity_Group']], on='asin', how='left')

# ==========================================
# 5. 保存所有划分结果，彻底固化实验环境
# ==========================================
train_path = 'Strict_Train_Interactions.csv'
val_path = 'Strict_Val_Interactions.csv'
test_path = 'Strict_Test_Interactions.csv'
pop_path = 'Strict_Train_Item_Popularity.csv'

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)
item_popularity.to_csv(pop_path, index=False)

print(f"\n✅ 数据已锁定并安全保存至:\n   - {train_path}\n   - {val_path}\n   - {test_path}\n   - {pop_path}")

#===========================================================================================================================================

# ==========================================
# 准备阶段：加载上一步生成的数据
# ==========================================
print("0. 正在加载上一阶段的 Train/Val/Test 数据...")

train_df = pd.read_csv('Strict_Train_Interactions.csv')
val_df = pd.read_csv('Strict_Val_Interactions.csv')
test_df = pd.read_csv('Strict_Test_Interactions.csv')
item_popularity = pd.read_csv('Strict_Train_Item_Popularity.csv')

# 基础检查
required_train_cols = {'reviewerID', 'asin', 'Level_2', 'Popularity_Group'}
required_eval_cols = {'reviewerID', 'asin', 'Level_2'}

assert required_train_cols.issubset(set(train_df.columns)), \
    f"train_df 缺少必要字段：{required_train_cols - set(train_df.columns)}"

assert required_eval_cols.issubset(set(val_df.columns)), \
    f"val_df 缺少必要字段：{required_eval_cols - set(val_df.columns)}"

assert required_eval_cols.issubset(set(test_df.columns)), \
    f"test_df 缺少必要字段：{required_eval_cols - set(test_df.columns)}"

assert {'asin', 'Popularity_Group'}.issubset(set(item_popularity.columns)), \
    "item_popularity 缺少 asin 或 Popularity_Group 列"

print("-> 数据加载完成。")


# ==========================================
# Step 1. 补全 val/test 的流行度标签，并过滤无效 item
# ==========================================
print("\n--- Step 1: 处理 Val / Test 流行度标签与未见 Item 过滤 ---")

def apply_popularity_and_filter(df, pop_df, name):
    """
    使用 train 中统计得到的流行度标签为 val/test 打标签，
    并过滤掉 train 中未出现过的 item。
    """
    initial_len = len(df)
    initial_users = df['reviewerID'].nunique()

    # 仅使用 train 统计得到的 pop_df 做映射
    df = pd.merge(df, pop_df[['asin', 'Popularity_Group']], on='asin', how='left')

    # 过滤掉 train 中未见过的 item
    df_filtered = df.dropna(subset=['Popularity_Group']).copy()

    final_len = len(df_filtered)
    final_users = df_filtered['reviewerID'].nunique()

    print(f"[{name}] 过滤前记录数: {initial_len}")
    print(f"[{name}] 过滤后记录数: {final_len}")
    print(f"[{name}] 剔除记录数  : {initial_len - final_len}")
    print(f"[{name}] 过滤前用户数: {initial_users}")
    print(f"[{name}] 过滤后用户数: {final_users}")
    print("-" * 50)

    return df_filtered

val_df = apply_popularity_and_filter(val_df, item_popularity, "Validation Set")
test_df = apply_popularity_and_filter(test_df, item_popularity, "Test Set")


# ==========================================
# Step 2. 基于 train_df 构建用户真实类目画像
# ==========================================
print("\n--- Step 2: 构建用户真实画像 true_matrix ---")

# 用户(行) x 类目(列) 的频次矩阵
true_matrix = pd.crosstab(train_df['reviewerID'], train_df['Level_2'])

# 频次 > 0 的位置设为 1，否则为 0
true_matrix = (true_matrix > 0).astype(int)

# 固定列顺序，确保后续所有矩阵维度一致
true_matrix = true_matrix.reindex(sorted(true_matrix.columns), axis=1)

print("-> true_matrix 构建完成！")
print(f"-> 矩阵形状（用户数 x 类目数）: {true_matrix.shape}")
print(f"-> Level_2 类目总数: {true_matrix.shape[1]}")
print("-> 前 3 行预览：")
print(true_matrix.head(3))


# ==========================================
# Step 3. 基于 train_df 构建自适应 epsilon 矩阵
# ==========================================
print("\n--- Step 3: 构建自适应隐私预算矩阵 eps_matrix ---")

EPS_DICT = {
    'Head': 4.0,
    'Torso': 2.0,
    'Tail': 0.5
}
DEFAULT_EPS = 1.6725

# 将流行度标签映射为 epsilon
train_df = train_df.copy()
train_df['epsilon'] = train_df['Popularity_Group'].map(EPS_DICT)

# 检查是否存在未成功映射的记录
missing_eps = train_df['epsilon'].isna().sum()
print(f"-> epsilon 映射完成，未成功映射记录数: {missing_eps}")
assert missing_eps == 0, "存在无法映射 epsilon 的 Popularity_Group，请检查标签内容"

# 对每个用户-类目，取最小 epsilon（最敏感的 item 决定保护强度）
eps_matrix = (
    train_df.groupby(['reviewerID', 'Level_2'])['epsilon']
    .min()
    .unstack()
    .reindex(index=true_matrix.index, columns=true_matrix.columns)
    .fillna(DEFAULT_EPS)
)

print("-> eps_matrix 构建完成！")
print(f"-> 矩阵形状: {eps_matrix.shape}")
print(f"-> eps_matrix 中 NaN 数量: {eps_matrix.isna().sum().sum()}")
print("-> 前 3 行预览：")
print(eps_matrix.head(3))


# ==========================================
# Step 4. 生成三种训练输入表示 (Randomized Response)
# ==========================================
print("\n--- Step 4: 执行 LDP 扰动，生成三种训练表示 ---")

def randomized_response_perturbation(X_true, eps_matrix_or_scalar):
    """
    通用随机响应函数
    参数：
        X_true: 二值 numpy array
        eps_matrix_or_scalar: 标量（固定预算）或与 X_true 同形状的 numpy array（自适应预算）
    返回：
        X_perturbed: 扰动后的二值 numpy array
    """
    # 保留真实值为 1 的概率 p
    p_mat = np.exp(eps_matrix_or_scalar) / (1.0 + np.exp(eps_matrix_or_scalar))
    # 当真实值为 0 时，翻转成 1 的概率 q
    q_mat = 1.0 / (1.0 + np.exp(eps_matrix_or_scalar))

    # 生成随机数矩阵
    rand_mat = np.random.uniform(low=0.0, high=1.0, size=X_true.shape)

    # 若真实值为1：以 p 概率保留1，否则变0
    # 若真实值为0：以 q 概率翻为1，否则保持0
    X_perturbed = np.where(
        X_true == 1,
        (rand_mat < p_mat).astype(int),
        (rand_mat < q_mat).astype(int)
    )
    return X_perturbed

# 转为 numpy，便于高效计算
X_true_val = true_matrix.values
eps_val = eps_matrix.values

# 1) No-noise version
X_true = true_matrix.copy()
print("1) X_true (No-noise version) 准备完毕。")

# 2) Fixed-budget perturbation version
np.random.seed(42)  # 固定随机种子，保证实验可复现
EPS_FIXED = 1.6725
X_fixed_val = randomized_response_perturbation(X_true_val, eps_matrix_or_scalar=EPS_FIXED)
X_fixed = pd.DataFrame(X_fixed_val, index=true_matrix.index, columns=true_matrix.columns)
print(f"2) X_fixed (Fixed-budget, epsilon={EPS_FIXED}) 生成完毕。")
print(f"   1 的总数变化: {X_true_val.sum()} -> {X_fixed_val.sum()}")

# 3) Adaptive-budget perturbation version
np.random.seed(42)  # 固定随机种子，保证实验可复现
X_adaptive_val = randomized_response_perturbation(X_true_val, eps_matrix_or_scalar=eps_val)
X_adaptive = pd.DataFrame(X_adaptive_val, index=true_matrix.index, columns=true_matrix.columns)
print("3) X_adaptive (Adaptive-budget) 生成完毕。")
print(f"   1 的总数变化: {X_true_val.sum()} -> {X_adaptive_val.sum()}")


# ==========================================
# Step 5. 输出和保存中间结果
# ==========================================
print("\n--- Step 5: 落盘保存数据 ---")

true_matrix.to_csv('true_matrix.csv', index=True)
eps_matrix.to_csv('eps_matrix.csv', index=True)
X_true.to_csv('X_true.csv', index=True)
X_fixed.to_csv('X_fixed.csv', index=True)
X_adaptive.to_csv('X_adaptive.csv', index=True)
val_df.to_csv('val_df_processed.csv', index=False)
test_df.to_csv('test_df_processed.csv', index=False)

print("✅ 所有文件已保存完成：")
print("   - true_matrix.csv")
print("   - eps_matrix.csv")
print("   - X_true.csv")
print("   - X_fixed.csv")
print("   - X_adaptive.csv")
print("   - val_df_processed.csv")
print("   - test_df_processed.csv")
print("\n🎉 训练输入构建阶段已完成，可以进入下一步：KMeans 聚类与 item-level 推荐评估。")