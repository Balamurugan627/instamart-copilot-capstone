import os
import sqlite3
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Task 2: Audit the raw data
# ---------------------------------------------------------
customers_raw = pd.read_csv("data/raw/customers.csv", keep_default_na=False)
products_raw = pd.read_csv("data/raw/products.csv", keep_default_na=False)
orders_raw = pd.read_csv("data/raw/orders.csv", keep_default_na=False)
tickets_raw = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)

print("--- Table Shapes & Missing Counts ---")
for name, df in [("customers", customers_raw), ("products", products_raw), ("orders", orders_raw), ("tickets", tickets_raw)]:
    print(f"\n{name} shape: {df.shape}")
    for col in df.columns:
        blank_cnt = (df[col].astype(str).str.strip() == "").sum()
        pct = (blank_cnt / len(df)) * 100
        print(f"  {col}: dtype={df[col].dtype}, missing={blank_cnt} ({pct:.2f}%)")

amt_num = pd.to_numeric(orders_raw["order_amount"], errors="coerce")
deliv_num = pd.to_numeric(orders_raw["delivery_minutes"], errors="coerce")
age_num = pd.to_numeric(customers_raw["age"], errors="coerce")

print("\n--- Audit Checkpoints ---")
print("Raw order rows:", len(orders_raw))
print("Exact duplicate rows:", orders_raw.duplicated().sum())
print("Pending rows:", (orders_raw["status"] == "pending").sum())
print("Distinct order_id:", orders_raw["order_id"].nunique())
print("Non-numeric order_amount:", amt_num.isna().sum())
print("Negative delivery_minutes:", (deliv_num < 0).sum())
print("Blank delivery_minutes:", (orders_raw["delivery_minutes"] == "").sum())
print("Blank age:", (customers_raw["age"] == "").sum())
print("Age outside 10-100:", ((age_num < 10) | (age_num > 100)).sum())
print("Raw city spellings:", customers_raw["city"].nunique())
print("Raw non-blank gender spellings:", customers_raw.loc[customers_raw["gender"] != "", "gender"].nunique())
print("Blank gender:", (customers_raw["gender"] == "").sum())

# ---------------------------------------------------------
# Task 3: Clean customers
# ---------------------------------------------------------
customers = pd.read_csv("data/raw/customers.csv", keep_default_na=False)

# (a) Standardise city
customers["city"] = customers["city"].str.strip().str.title()

# (b) Standardise gender
g_map = {"m": "male", "male": "male", "f": "female", "female": "female", "": "unknown"}
customers["gender"] = customers["gender"].str.strip().str.lower().map(g_map).fillna("unknown")

# (c) Clean age and fill with city median
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers.loc[(customers["age"] < 10) | (customers["age"] > 100), "age"] = np.nan
customers["age"] = customers["age"].fillna(customers.groupby("city")["age"].transform("median"))

# (d) Convert signup_date to datetime
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

print("\n--- Cleaned Customers Summary ---")
print(customers["city"].value_counts())
print(customers["gender"].value_counts())
print("Missing ages:", customers["age"].isna().sum(), "| Min age:", customers["age"].min(), "| Max age:", customers["age"].max())

# ---------------------------------------------------------
# Task 4: Clean orders
# ---------------------------------------------------------
orders = pd.read_csv("data/raw/orders.csv", keep_default_na=False)
products = pd.read_csv("data/raw/products.csv")

# 1. Convert order_ts to datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# 2. Remove exact duplicates
orders = orders.drop_duplicates().copy()

# 3. Keep latest record per order_id
orders = orders.sort_values("order_ts").drop_duplicates(subset=["order_id"], keep="last").sort_index().copy()

# 4. Convert order_amount to numeric and fill missing from product price
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
orders = orders.merge(products[["product_id", "price_inr", "category"]], on="product_id", how="left")

calc_amt = (orders["price_inr"] * orders["quantity"] * (1 - orders["discount_pct"] / 100.0)).round(2)
orders["order_amount"] = orders["order_amount"].fillna(calc_amt)

# 5. Clean delivery_minutes and impute completed orders with category median
orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan

comp_mask = orders["status"] == "completed"
cat_medians = orders.loc[comp_mask].groupby("category")["delivery_minutes"].transform("median")
orders.loc[comp_mask, "delivery_minutes"] = orders.loc[comp_mask, "delivery_minutes"].fillna(cat_medians)

# Drop helper columns from products merge
orders = orders.drop(columns=["price_inr", "category"])

print("\n--- Cleaned Orders Summary ---")
print("Total orders:", len(orders))
print(orders["status"].value_counts())
print("Completed revenue (INR):", round(orders.loc[comp_mask, "order_amount"].sum(), 2))

# ---------------------------------------------------------
# Task 5: Store clean tables to CSV and SQLite
# ---------------------------------------------------------
os.makedirs("data/clean", exist_ok=True)

customers_out = customers.copy()
customers_out["signup_date"] = customers_out["signup_date"].dt.strftime("%Y-%m-%d")

orders_out = orders.copy()
orders_out["order_ts"] = orders_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

customers_out.to_csv("data/clean/customers.csv", index=False)
products.to_csv("data/clean/products.csv", index=False)
orders_out.to_csv("data/clean/orders.csv", index=False)

conn = sqlite3.connect("data/instamart.db")
customers_out.to_sql("customers", conn, if_exists="replace", index=False)
products.to_sql("products", conn, if_exists="replace", index=False)
orders_out.to_sql("orders", conn, if_exists="replace", index=False)
conn.close()

print("\nSaved clean CSVs to data/clean/ and SQLite tables to data/instamart.db")