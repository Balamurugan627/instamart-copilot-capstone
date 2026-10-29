"""part_b_model.py -- Part B: Customer Segmentation (RFM + KMeans) and Churn Prediction."""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

CLEAN_DIR = "data/clean"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("--- Loading Clean Data ---")
customers = pd.read_csv(os.path.join(CLEAN_DIR, "customers.csv"))
orders = pd.read_csv(os.path.join(CLEAN_DIR, "orders.csv"))

# Filter only completed orders for RFM calculation
completed = orders[orders["status"] == "completed"].copy()
completed["order_ts"] = pd.to_datetime(completed["order_ts"])

# Reference date for recency (set to the day after the last order in the dataset)
ref_date = completed["order_ts"].max() + pd.Timedelta(days=1)

print("--- Engineering RFM Features ---")
rfm = completed.groupby("customer_id").agg(
    recency_days=("order_ts", lambda x: (ref_date - x.max()).days),
    frequency=("order_id", "count"),
    monetary=("order_amount", "sum")
).reset_index()

# Merge back with all customers (including those with 0 completed orders)
cust_rfm = customers.merge(rfm, on="customer_id", how="left")
cust_rfm["recency_days"] = cust_rfm["recency_days"].fillna(365)  # High recency for inactive
cust_rfm["frequency"] = cust_rfm["frequency"].fillna(0)
cust_rfm["monetary"] = cust_rfm["monetary"].fillna(0.0)

# Define churn / lapse target: active if ordered within last 60 days of the window, else churned/lapsed
cust_rfm["is_churned"] = (cust_rfm["recency_days"] > 60).astype(int)

print(f"Customer feature matrix shape: {cust_rfm.shape}")
print(f"Churn distribution:\n{cust_rfm['is_churned'].value_counts()}")

# --- K-Means Clustering ---
print("\n--- Running K-Means Segmentation ---")
feature_cols = ["recency_days", "frequency", "monetary"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cust_rfm[feature_cols])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cust_rfm["cluster"] = kmeans.fit_predict(X_scaled)
print("Cluster sizes:\n", cust_rfm["cluster"].value_counts())

# --- Churn Prediction Model ---
print("\n--- Training Churn Prediction Model ---")
X = cust_rfm[["recency_days", "frequency", "monetary", "age"]]
y = cust_rfm["is_churned"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, preds))
print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")

# Save models and features
joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_segmenter.pkl"))
joblib.dump(clf, os.path.join(MODEL_DIR, "churn_classifier.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "rfm_scaler.pkl"))
cust_rfm.to_csv("data/clean/customer_segments.csv", index=False)

print("\nPart B models trained successfully and saved to models/ and data/clean/customer_segments.csv!")