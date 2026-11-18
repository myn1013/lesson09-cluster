"""
生成本课两个数据集 + bank_customers.csv（实践指南版）
===================================================
1. mall_customers_sim.csv   — 300行7列，两维肉眼三块（income-balance），k=3 跑出 125/125/50
2. mall_customers_yuan.csv  — 同上，balance 以元计（量纲陷阱版）
3. bank_customers.csv       — 200行，含四特征 + 扩展特征（实践指南作业用）
随机种子固定，保证可复现
"""
import numpy as np
import pandas as pd

np.random.seed(42)

genders = ["男", "女"]
cities = ["昆明", "曲靖", "玉溪", "大理"]


# ============================================================
# 1 & 2. mall_customers_sim / mall_customers_yuan (k=3)
# ============================================================
clusters_mall = [
    # (人数, income均值/方差, balance均值/方差, age均值/方差, prod均值/方差)
    (125, (15, 3**2), (8, 2**2),  (28, 3**2), (1.5, 0.3**2)),
    (125, (30, 4**2), (25, 5**2), (38, 4**2), (3.0, 0.5**2)),
    (50,  (45, 4**2), (48, 6**2), (48, 5**2), (5.0, 0.8**2)),
]

rows = []
cid = 1
for n, (im, iv), (bm, bv), (am, av), (pm, pv) in clusters_mall:
    for _ in range(n):
        rows.append([
            f"C{cid:04d}",
            int(np.clip(np.random.normal(am, av**0.5), 18, 70)),
            round(np.clip(np.random.normal(im, iv**0.5), 3, 60), 2),
            round(np.clip(np.random.normal(bm, bv**0.5), 1, 80), 2),
            int(np.clip(np.random.normal(pm, pv**0.5), 1, 8)),
            np.random.choice(genders),
            np.random.choice(cities),
        ])
        cid += 1

df_mall = pd.DataFrame(rows, columns=[
    "customer_id", "age", "income", "balance",
    "product_count", "gender", "city"])
df_mall = df_mall.sample(frac=1, random_state=42).reset_index(drop=True)

df_mall.to_csv("data/mall_customers_sim.csv", index=False, encoding="utf-8-sig")

df_mall_yuan = df_mall.copy()
df_mall_yuan["balance"] = (df_mall_yuan["balance"] * 10000).round(0).astype(int)
df_mall_yuan.to_csv("data/mall_customers_yuan.csv", index=False, encoding="utf-8-sig")

print(f"✅ mall_customers_sim.csv   : {df_mall.shape}, "
      f"income {df_mall.income.min():.1f}~{df_mall.income.max():.1f}, "
      f"balance(万元) {df_mall.balance.min():.1f}~{df_mall.balance.max():.1f}")
print(f"✅ mall_customers_yuan.csv  : balance(元) {df_mall_yuan.balance.min()}~{df_mall_yuan.balance.max()}")


# ============================================================
# 3. bank_customers.csv (实践指南版，k=4 四特征，200客户)
#    四簇结构：
#      簇0 低价值沉睡客群 ~55人  — balance低 消费低 产品少 逾期多
#      簇1 潜力客群       ~50人  — 余额中 消费上升 产品中 逾期少
#      簇2 主力客群       ~60人  — 余额中高 消费中高 产品多 逾期极少
#      簇3 高净值VIP      ~35人  — 余额高 消费高 产品多 几乎不逾期
# ============================================================
np.random.seed(123)   # 换一个种子，两数据集独立

clusters_bank = [
    # (人数, balance(万), avg_monthly_spend(万), product_count, overdue_count, account_age_months, age, income(万))
    (55, (5, 2),  (0.5, 0.2), (1.2, 0.3), (3.5, 1.0), (18, 6),  (28, 4), (12, 3)),
    (50, (18, 4), (2.0, 0.5), (2.8, 0.5), (0.8, 0.4), (36, 8),  (35, 5), (25, 4)),
    (60, (32, 6), (3.5, 0.8), (4.2, 0.8), (0.2, 0.2), (60, 10), (42, 5), (38, 5)),
    (35, (55, 8), (6.0, 1.2), (5.5, 1.0), (0.05, 0.1), (72, 12), (50, 6), (55, 6)),
]

rows = []
cid = 1
for n, bal, spend, prod, overdue, age_m, age, inc in clusters_bank:
    for _ in range(n):
        rows.append([
            f"B{cid:04d}",
            int(np.clip(np.random.normal(age[0], age[1]), 20, 70)),
            round(np.clip(np.random.normal(inc[0], inc[1]), 2, 70), 2),
            round(np.clip(np.random.normal(bal[0], bal[1]), 0.5, 90), 2),
            round(np.clip(np.random.normal(spend[0], spend[1]), 0.1, 12), 2),
            int(np.clip(np.random.normal(prod[0], prod[1]), 1, 8)),
            round(np.clip(np.random.normal(overdue[0], overdue[1]), 0, 8), 1),
            int(np.clip(np.random.normal(age_m[0], age_m[1]), 6, 120)),
            np.random.choice(genders),
            np.random.choice(cities),
        ])
        cid += 1

df_bank = pd.DataFrame(rows, columns=[
    "customer_id", "age", "income", "balance",
    "avg_monthly_spend", "product_count", "overdue_count",
    "account_age_months", "gender", "city"])
df_bank = df_bank.sample(frac=1, random_state=123).reset_index(drop=True)
df_bank.to_csv("data/bank_customers.csv", index=False, encoding="utf-8-sig")

print(f"✅ bank_customers.csv       : {df_bank.shape}")
print(f"   balance 范围 {df_bank.balance.min():.1f}~{df_bank.balance.max():.1f} 万元")
print(f"   avg_monthly_spend 范围 {df_bank.avg_monthly_spend.min():.1f}~{df_bank.avg_monthly_spend.max():.1f} 万元")
print(f"   product_count 范围 {df_bank.product_count.min()}~{df_bank.product_count.max()}")
print(f"   overdue_count 范围 {df_bank.overdue_count.min()}~{df_bank.overdue_count.max()}")
print(f"   account_age_months 范围 {df_bank.account_age_months.min()}~{df_bank.account_age_months.max()}")
