"""
案例复现三步 + 量纲陷阱对比
===========================
Step 1  聚类跑通：bank_customers.csv 四特征标准化后 KMeans(k=4)
Step 2  肘部法则：k=1..9 扫 SSE + 轮廓系数
Step 3  分群画像：groupby 均值表 + 中文业务命名 + 雷达图
Bonus   量纲陷阱：mall_customers_yuan.csv 不标准化直接跑的翻车对比
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------- 字体：Windows 下用微软雅黑 ----------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

RANDOM_STATE = 42

# ================================================================
# 读取 bank_customers.csv，选取四特征
# ================================================================
FEATS = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]

df = pd.read_csv("data/bank_customers.csv")
print("=" * 60)
print("【数据初检】bank_customers.csv")
print(f"行列数：{df.shape}")
print("缺失值：\n", df.isnull().sum())
print("\n数值列基本统计：")
print(df[FEATS + ["age", "income", "account_age_months"]].describe())

# ---------- 初始散点图（balance vs avg_monthly_spend） ----------
fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(df["balance"], df["avg_monthly_spend"],
                s=df["product_count"] * 20 + 30, alpha=0.6,
                c=df["account_age_months"], cmap="viridis")
ax.set_xlabel("账户平均余额（万元）")
ax.set_ylabel("月均消费（万元）")
ax.set_title("01 客户初始散点图（点大小=持有产品数，颜色=账户年龄）")
plt.colorbar(sc, label="账户年龄（月）")
plt.tight_layout()
plt.savefig("outputs/01_raw_scatter.png", dpi=150)
plt.close()
print("\n✅ outputs/01_raw_scatter.png 已保存")


# ================================================================
# Step 1  聚类跑通（四特征 + 标准化 + KMeans k=4）
# ================================================================
print("\n" + "=" * 60)
print("【Step 1】KMeans 聚类 k=4")

X = df[FEATS].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

km = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
df["cluster"] = km.fit_predict(X_scaled)

# 检查点1：四簇人数
counts = df["cluster"].value_counts().sort_index()
print("\n检查点 1 —— 四簇人数：")
for c, n in counts.items():
    print(f"  簇{c}：{n} 人（{n/len(df)*100:.1f}%）")


# ================================================================
# Step 2  肘部法则 + 轮廓系数
# ================================================================
print("\n" + "=" * 60)
print("【Step 2】选 k —— 肘部法 + 轮廓系数")

k_range = range(1, 10)
sses, silhous = [], []
for k in k_range:
    m = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_scaled)
    sses.append(m.inertia_)
    if k >= 2:
        silhous.append(silhouette_score(X_scaled, m.labels_))
    else:
        silhous.append(np.nan)

print("\nk | SSE        | 轮廓系数")
print("-" * 32)
for i, k in enumerate(k_range):
    print(f"{k} | {sses[i]:>10.2f} | {silhous[i]:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(k_range, sses, "o-", color="#2E86AB", lw=2)
ax1.axvline(x=4, color="red", linestyle="--", label="肘部拐点候选 k=4")
ax1.set_xlabel("簇数 k")
ax1.set_ylabel("簇内平方和 SSE")
ax1.set_title("肘部法则 —— SSE 随 k 变化")
ax1.legend()

ax2.plot(range(2, 10), silhous[1:], "s-", color="#A23B72", lw=2)
best_k = range(2, 10)[np.argmax(silhous[1:])]
ax2.axvline(x=best_k, color="red", linestyle="--", label=f"轮廓系数最高 k={best_k}")
ax2.set_xlabel("簇数 k")
ax2.set_ylabel("轮廓系数")
ax2.set_title("轮廓系数 —— 越高越好")
ax2.legend()

plt.tight_layout()
plt.savefig("outputs/02_choose_k.png", dpi=150)
plt.close()
print(f"\n✅ outputs/02_choose_k.png 已保存")
print(f"   肘部法建议 k=4；轮廓系数最高 k={best_k}")
print(f"   综合业务需求（四套客户打法），取 k=4")


# ================================================================
# Step 3  分群画像 + 中文命名 + 雷达图
# ================================================================
print("\n" + "=" * 60)
print("【Step 3】分群画像")

profile = df.groupby("cluster")[FEATS + ["age", "income", "account_age_months"]].mean().round(2)
print("\n各簇特征均值画像表：")
print(profile.to_string())

# ---------- 中文业务命名 ----------
# 根据均值排序自动起名字
mean_balance = profile["balance"]
mean_spend   = profile["avg_monthly_spend"]
mean_overdue = profile["overdue_count"]
mean_prod    = profile["product_count"]
mean_age_m   = profile["account_age_months"]

names = {}
for c in profile.index:
    bal, sp, ov, pr, am = mean_balance[c], mean_spend[c], mean_overdue[c], mean_prod[c], mean_age_m[c]
    if bal > 45 and sp > 4 and pr > 4 and ov < 0.2:
        names[c] = "高净值VIP客群"
    elif bal > 25 and sp > 2.5 and pr > 3 and ov < 0.5:
        names[c] = "主力活跃客群"
    elif bal < 12 and sp < 1 and pr < 2 and ov > 2:
        names[c] = "低价值沉睡客群"
    else:
        names[c] = "潜力成长客群"

df["cluster_name"] = df["cluster"].map(names)
print("\n四簇中文业务命名：")
for c, nm in names.items():
    print(f"  簇{c} → {nm}")


# ---------- 雷达图 ----------
# 用标准化后的均值
scaled_means = scaler.transform(df.groupby("cluster")[FEATS].mean())
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2 * np.pi, len(FEATS), endpoint=False).tolist()
angles += angles[:1]

colors = ["#FF6B6B", "#4ECDC4", "#FFD93D", "#6BCB77"]
for i, c in enumerate(sorted(df["cluster"].unique())):
    vals = scaled_means[i].tolist() + [scaled_means[i][0]]
    ax.plot(angles, vals, "o-", lw=2, label=f"簇{c} · {names[c]}", color=colors[i])
    ax.fill(angles, vals, alpha=0.12, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(["余额", "月均消费", "产品数", "逾期次数"])
ax.set_title("四簇画像雷达图（标准化空间）", y=1.08)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
plt.tight_layout()
plt.savefig("outputs/03_radar_profile.png", dpi=150)
plt.close()
print("✅ outputs/03_radar_profile.png 已保存")


# ---------- 聚类着色散点图 ----------
fig, ax = plt.subplots(figsize=(10, 7))
for c in sorted(df["cluster"].unique()):
    sub = df[df["cluster"] == c]
    ax.scatter(sub["balance"], sub["avg_monthly_spend"],
               s=sub["product_count"] * 20 + 30, alpha=0.6,
               color=colors[c], label=f"簇{c} {names[c]} ({len(sub)}人)")
# 质心画星形
centers_orig = scaler.inverse_transform(km.cluster_centers_)
ax.scatter(centers_orig[:, 0], centers_orig[:, 1],
           marker="*", s=400, c="black", edgecolors="white", zorder=5)
ax.set_xlabel("账户平均余额（万元）")
ax.set_ylabel("月均消费（万元）")
ax.set_title("03 客户分群着色散点图（k=4）")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/04_clustered_scatter.png", dpi=150)
plt.close()
print("✅ outputs/04_clustered_scatter.png 已保存")


# ---------- 量纲陷阱对比 ----------
print("\n" + "=" * 60)
print("【Bonus】量纲陷阱演示（mall_customers_yuan.csv）")

df_yuan = pd.read_csv("data/mall_customers_yuan.csv")
feats2 = ["income", "balance"]

# 不标准化 —— 翻车
X_raw = df_yuan[feats2].values
km_raw = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
df_yuan["cluster_raw"] = km_raw.fit_predict(X_raw)
raw_cnt = df_yuan["cluster_raw"].value_counts().sort_index().values

# 标准化 —— 修正
X_s2 = StandardScaler().fit_transform(X_raw)
km_ok = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
df_yuan["cluster_ok"] = km_ok.fit_predict(X_s2)
ok_cnt = df_yuan["cluster_ok"].value_counts().sort_index().values

print(f"不标准化（翻车）：三簇人数 {raw_cnt.tolist()}")
print(f"标准化后（修正）：三簇人数 {ok_cnt.tolist()}")
print(f"balance 以元计，最大值 {df_yuan.balance.max()}，income 以万元计，最大值 {df_yuan.income.max()}")
print(f"→ 两列差 4 个数量级，balance 完全淹没 income")

# 并排对比图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
for lbl, col in enumerate(sorted(df_yuan["cluster_raw"].unique())):
    sub = df_yuan[df_yuan["cluster_raw"] == col]
    ax1.scatter(sub["income"], sub["balance"], label=f"簇{col} ({len(sub)}人)", alpha=0.7)
ax1.set_xlabel("income（万元）"); ax1.set_ylabel("balance（元）")
ax1.set_title("不标准化 —— 量纲陷阱")
ax1.legend()

for lbl, col in enumerate(sorted(df_yuan["cluster_ok"].unique())):
    sub = df_yuan[df_yuan["cluster_ok"] == col]
    ax2.scatter(sub["income"], sub["balance"], label=f"簇{col} ({len(sub)}人)", alpha=0.7)
ax2.set_xlabel("income（万元）"); ax2.set_ylabel("balance（元）")
ax2.set_title("标准化后 —— 修正正确")
ax2.legend()

plt.tight_layout()
plt.savefig("outputs/05_scale_compare.png", dpi=150)
plt.close()
print("✅ outputs/05_scale_compare.png 已保存")


# ================================================================
# 输出摘要
# ================================================================
print("\n" + "=" * 60)
print("【三问验证】")
print("① 初始散点图肉眼可见约 3~4 个自然块")
print(f"② k=4 四簇人数：{counts.values.tolist()}")
print(f"③ 四簇业务命名：{names}")
print("\n✅ 案例复现三步全部完成！")
