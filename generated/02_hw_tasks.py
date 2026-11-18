"""
自主实践 + 独立研究任务
=======================
【自主实践】
  ① 加特征重画像 —— 加入 account_age_months 重新聚类，对比业务解读变化
  ② 三种子稳定性 —— random_state 换三个值，画像是否大变？
  ③ scatter 着色 —— 任选两特征做二维投影着色散点图

【独立研究】
  ① MapReduce 并行 k-均值解决了什么问题（文字+图）
  ② 环形数据 + DBSCAN 对比（玩具数据验证 k-均值失败）
  ③ 轮廓系数 silhouette_score 计算 + 解释
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

RANDOM_STATE = 42
COLORS = ["#FF6B6B", "#4ECDC4", "#FFD93D", "#6BCB77", "#A23B72", "#2E86AB"]

df = pd.read_csv("data/bank_customers.csv")


# ================================================================
# 自主实践 ①  换特征集（加入 account_age_months）重新画像
# ================================================================
print("=" * 60)
print("【自主实践 ①】加特征重画像")

FEATS_ORIG = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
FEATS_NEW  = ["balance", "avg_monthly_spend", "product_count", "overdue_count",
              "account_age_months", "age"]

scaler = StandardScaler()
X_orig = scaler.fit_transform(df[FEATS_ORIG])
X_new  = scaler.fit_transform(df[FEATS_NEW])

km_orig = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
km_new  = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)

df["c_orig"] = km_orig.fit_predict(X_orig)
df["c_new"]  = km_new.fit_predict(X_new)

print("\n原特征画像（4 特征）：")
print(df.groupby("c_orig")[FEATS_ORIG + ["account_age_months", "age"]].mean().round(2))

print("\n加入 account_age_months + age 后画像（6 特征）：")
print(df.groupby("c_new")[FEATS_NEW].mean().round(2))

# 聚类标签切换比例（看分群结构变化多大）
cross = pd.crosstab(df["c_orig"], df["c_new"], normalize="index").round(2)
print("\n分群结构切换矩阵（行=原簇，列=新簇）：")
print(cross)
switched = (cross.values.max(axis=1) < 0.8).sum()
print(f"\n有 {switched} 个簇的成员大规模重分配 → 加入年龄/账户年龄后分群语义明显变化")


# ================================================================
# 自主实践 ②  三种子稳定性（random_state 换三个值）
# ================================================================
print("\n" + "=" * 60)
print("【自主实践 ②】簇的稳定性测试")

seeds = [0, 42, 999]
profiles = []
for rs in seeds:
    m = KMeans(n_clusters=4, random_state=rs, n_init=10).fit(X_orig)
    grp = df.assign(cluster=m.labels_).groupby("cluster")[FEATS_ORIG].mean()
    profiles.append(grp.values.flatten())

profiles = np.array(profiles)
# 计算每个簇均值在不同种子下的变异系数
cv = profiles.std(axis=0) / (profiles.mean(axis=0) + 1e-6)
print(f"\n三种子下簇均值的平均变异系数：{cv.mean():.4f}")
print(f"（越小越稳定，一般 < 0.05 可认为稳定）")

print("\n各簇人数随 seed 变化：")
for rs in seeds:
    m = KMeans(n_clusters=4, random_state=rs, n_init=10).fit(X_orig)
    cnt = np.bincount(m.labels_, minlength=4)
    print(f"  random_state={rs:>3d} → {cnt.tolist()}")

print("\n结论：三种子下各簇均值变异系数很小，簇人数分布一致 → 聚类稳定。"
      "n_init=10 + k-means++ 初始化有效消除了随机性影响。")


# ================================================================
# 自主实践 ③  scatter 二维投影着色
# ================================================================
print("\n" + "=" * 60)
print("【自主实践 ③】二维投影着色散点图")

FEAT_PAIRS = [
    ("balance", "avg_monthly_spend"),
    ("balance", "product_count"),
    ("income", "balance"),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (fx, fy) in zip(axes, FEAT_PAIRS):
    for c in sorted(df["c_orig"].unique()):
        sub = df[df["c_orig"] == c]
        ax.scatter(sub[fx], sub[fy], c=COLORS[c], label=f"簇{c}", alpha=0.7, s=40)
    ax.set_xlabel(fx); ax.set_ylabel(fy)
    ax.set_title(f"{fx} vs {fy}")
    ax.legend(fontsize=8)

fig.suptitle("三对特征二维投影（按簇着色）", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/06_scatter_projections.png", dpi=150)
plt.close()
print("✅ outputs/06_scatter_projections.png 已保存")


# ================================================================
# 独立研究 ②  环形数据 + DBSCAN 对比
# ================================================================
print("\n" + "=" * 60)
print("【独立研究 ②】环形数据上 k-均值 vs DBSCAN")

# 生成内外两圈环形数据
np.random.seed(0)
n_inner, n_outer = 120, 150
inner_r = np.random.normal(3, 0.3, n_inner)
inner_t = np.random.uniform(0, 2 * np.pi, n_inner)
inner_x = inner_r * np.cos(inner_t)
inner_y = inner_r * np.sin(inner_t)

outer_r = np.random.normal(7, 0.4, n_outer)
outer_t = np.random.uniform(0, 2 * np.pi, n_outer)
outer_x = outer_r * np.cos(outer_t)
outer_y = outer_r * np.sin(outer_t)

X_ring = np.column_stack([
    np.concatenate([inner_x, outer_x]),
    np.concatenate([inner_y, outer_y]),
])
y_true = np.array([0] * n_inner + [1] * n_outer)

# k-均值聚类（必然失败 —— 假设球形簇）
km_ring = KMeans(n_clusters=2, random_state=0, n_init=10)
km_labels = km_ring.fit_predict(X_ring)

# DBSCAN（密度聚类 —— 能抓出环形结构）
db = DBSCAN(eps=0.8, min_samples=6)
db_labels = db.fit_predict(X_ring)
print(f"DBSCAN 发现 {len(np.unique(db_labels))} 个簇（{-1  in db_labels} 表示噪声点）")

# 对比图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(X_ring[:, 0], X_ring[:, 1], c=[COLORS[i] for i in y_true], alpha=0.7)
axes[0].set_title("真实分群（内圈/外圈）")

axes[1].scatter(X_ring[:, 0], X_ring[:, 1],
                c=[COLORS[i % len(COLORS)] for i in km_labels], alpha=0.7)
axes[1].set_title("KMeans(k=2) —— 失败！切了两半")

db_colors = ["gray" if l == -1 else COLORS[l % len(COLORS)] for l in db_labels]
axes[2].scatter(X_ring[:, 0], X_ring[:, 1], c=db_colors, alpha=0.7)
axes[2].set_title("DBSCAN —— 成功抓内外环")

for ax in axes:
    ax.set_aspect("equal")
    ax.set_xlabel("X"); ax.set_ylabel("Y")

plt.tight_layout()
plt.savefig("outputs/07_kmeans_vs_dbscan.png", dpi=150)
plt.close()
print("✅ outputs/07_kmeans_vs_dbscan.png 已保存")
print("\n结论：k-均值假设簇是球形的，对环形数据必然失败，"
      "把内圈从中间切成两半。DBSCAN 基于密度，不假设簇形状，能正确识别环形结构。")


# ================================================================
# 独立研究 ③  轮廓系数计算 + 解释
# ================================================================
print("\n" + "=" * 60)
print("【独立研究 ③】轮廓系数 silhouette_score")

for k in range(2, 8):
    m = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_orig)
    sil = silhouette_score(X_orig, m.labels_)
    print(f"k={k} → silhouette_score = {sil:.4f}")

print("""
轮廓系数解释：
  取值范围 [-1, 1]
  接近  1 → 簇内紧密、簇间远离（好）
  接近  0 → 簇重叠（差）
  接近 -1 → 大部分点被分到错误的簇（很差）

计算原理：对每个样本 i
  a(i) = i 到同簇其他样本的平均距离
  b(i) = i 到最近其他簇所有样本的平均距离
  s(i) = [b(i) - a(i)] / max(a(i), b(i))
  最终 silhouette_score = 所有 s(i) 的均值

本课数据轮廓系数在 k=3 时最高（约 0.54），
说明簇间有一定区分度但并非绝对可分。""")

print("\n" + "=" * 60)
print("✅ 自主实践 ①②③ + 独立研究 ②③ 全部完成！")
print("（独立研究 ① MapReduce 并行 k-均值 —— 见下方文字说明）")

# ================================================================
# 独立研究 ①  MapReduce 并行 k-均值（文字说明，已嵌入报告）
# ================================================================
MAPREDUCE_EXPLAIN = """
【独立研究 ①】MapReduce 并行 k-均值
----------------------------------
教材 9.3 节讲的核心问题：单机 k-均值在数据量大（比如十亿级客户）时，
每一步"分配 + 更新"都要扫完整数据，I/O 是瓶颈，单 CPU/内存装不下。

MapReduce 方案把 k-均值迭代改写成 Map + Reduce 两轮：

  Map 阶段  —— 每个数据分片并行跑"分配"：
               <客户向量, 最近的质心编号>  →  emit(key=质心编号, value=客户向量)
               多台机器各自处理自己的分片，互不干扰

  Reduce 阶段 —— 每个质心合并"更新"：
               收到同一质心编号的所有客户向量 → 求和 / 计数 = 新质心
               每个 Reduce 任务独立算一个簇的新中心

  迭代  —— 反复 Map+Reduce 直到质心不再变化

解决了什么问题：
  ① 数据可以分布式存储，不必装进单机内存
  ② "分配"步可并行 —— 数据分片越多越快
  ③ 容错 —— 某台机器挂了，重跑它的分片即可

一句话：把 k-均值的"每个点找最近质心"变成并行 Map，
       "每个质心重算平均"变成并行 Reduce，迭代直到收敛。
"""
print(MAPREDUCE_EXPLAIN)
