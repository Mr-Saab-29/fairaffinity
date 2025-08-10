from pathlib import Path
import pandas as pd

d = Path("data/interim")

dfs = {
    "clients": pd.read_parquet(d/"clients.parquet"),
    "products": pd.read_parquet(d/"products.parquet"),
    "stores": pd.read_parquet(d/"stores.parquet"),
    "transactions": pd.read_parquet(d/"transactions.parquet"),
    "stocks": pd.read_parquet(d/"stocks.parquet"),
}

#--------- Basic Shapes and Nulls ----------------
for name, df in dfs.items():
    print(f"\n[{name}] rows={len(df):,}, cols={df.shape[1]}")
    print(df.dtypes)
    print("null %:\n", (df.isna().mean()*100).round(2).sort_values(ascending=False).head(10))

# --------- Date Sanity Check ----------------
tx = dfs["transactions"]
print("\n[transactions] date range:", tx["SaleTransactionDate"].min(), "→", tx["SaleTransactionDate"].max())

#-------- Numeric Sanity Check --------------
bad_qty = (tx["Quantity"] < 0).sum()
bad_amt = (tx["SalesNetAmountEuro"] < 0).sum()
print(f"\n[transactions] negative qty: {bad_qty}, negative amount: {bad_amt}")

# --------- Key Integrity Check --------------
clients = dfs["clients"]["ClientID"].dropna().astype("Int64")
products = dfs["products"]["ProductID"].dropna().astype("Int64")
stores = dfs["stores"]["StoreID"].dropna().astype("Int64")

tx_bad_clients = (~tx["ClientID"].isin(clients)).sum()
tx_bad_products = (~tx["ProductID"].isin(products)).sum()
tx_bad_stores = (~tx["StoreID"].isin(stores)).sum()
print(f"\n[transactions] orphan keys -> clients:{tx_bad_clients}, products:{tx_bad_products}, stores:{tx_bad_stores}")

# --------- Stock Integrity Check --------------
stocks = dfs["stocks"]
stock_missing_products = (~stocks["ProductID"].isin(products)).sum()
print(f"[stocks] product ids not in products: {stock_missing_products}")

# --------- Categorical Samples --------------
print("\n[clients] gender value counts (top 5):")
print(dfs["clients"]["ClientGender"].value_counts(dropna=False).head())

print("\n[products] category samples:")
print(dfs["products"]["Category"].dropna().unique()[:10])