from pathlib import Path
import pandas as pd

#Paths
ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

#Building Interactions
def build_interactions():
    #Load canonical parquet data
    clients = pd.read_parquet(INTERIM / "clients.parquet")
    products = pd.read_parquet(INTERIM / "products.parquet")
    transactions = pd.read_parquet(INTERIM / "transactions.parquet")
    stores = pd.read_parquet(INTERIM / "stores.parquet")
    stocks = pd.read_parquet(INTERIM / "stocks.parquet")

    # Precompute Stock availability by (StoreCountry, ProductID)
    # Some files have Quantity as float; Treat NaN as 0 and sum

    stocks["Quantity"] = pd.to_numeric(stocks["Quantity"], errors="coerce").fillna(0.0)

    valid_pids = set(products["ProductID"].unique())
    stocks = stocks[stocks["ProductID"].isin(valid_pids)].copy()

    stocks_agg = (
        stocks.groupby(["StoreCountry", "ProductID"], as_index=False)['Quantity']
        .sum()
        .rename(columns={"Quantity": "StockQuantity"})
    )

    stocks_agg['Available'] = (stocks_agg['StockQuantity'] > 0).astype("int8")

    #Base joins
    df = transactions.merge(clients, on="ClientID", how="left", validate = "m:1")
    df = df.merge(products, on="ProductID", how="left", validate = "m:1")
    df = df.merge(stores, on="StoreID", how="left", validate = "m:1")

    # Temporal Features
    df['txn_date'] = pd.to_datetime(df['SaleTransactionDate'], errors='coerce')
    df['txn_year'] = df['txn_date'].dt.year
    df['txn_month'] = df['txn_date'].dt.month
    df['txn_dow'] = df['txn_date'].dt.dayofweek # 0 = Monday, 6 = Sunday
    df['txn_week'] = df['txn_date'].dt.isocalendar().week.astype("int64")
    df['is_weekend'] = df['txn_dow'].isin([5, 6]).astype("int8")  # Saturday/Sunday

    # ---------- Availabiltiy Flags ----------------
    # 1. Availability in the store country
    df = df.merge(
        stocks_agg[['StoreCountry', 'ProductID', 'Available']]
        .rename(columns={'Available': 'AvailableInStoreCountry'}),
        on =['StoreCountry', 'ProductID'],
        how='left', 
        validate='m:1',
    )

    # 2. Availability in the client's country
    stock_client_country = stocks_agg.rename(columns = {"StoreCountry" : "ClientCountry"})
    df = df.merge(
        stock_client_country[["ClientCountry", "ProductID", "Available"]]
        .rename(columns={'Available': 'AvailableInClientCountry'}),
        on=['ClientCountry', 'ProductID'],
        how='left',
        validate='m:1',
    )

    # Fill missing availability with flags with 0
    for col in ['AvailableInStoreCountry', 'AvailableInClientCountry']:
        df[col] = df[col].fillna(0).astype("int8")
    
    #Rerorder columns for readability
    col_order = [
    "ClientID", "ProductID", "SaleTransactionDate", "txn_date",
    "StoreID", "StoreCountry",
    "Quantity", "SalesNetAmountEuro",
    "ClientSegment", "ClientCountry", "ClientOptINEmail", "ClientOptINPhone",
    "ClientGender", "Age",
    "Category", "FamilyLevel1", "FamilyLevel2", "Universe",
    "txn_year", "txn_month", "txn_dow", "txn_week", "is_weekend",
    "AvailableInStoreCountry", "AvailableInClientCountry",
    ]
    # Ensure all columns in col_order are present in df
    df = df[[c for c in col_order if c in df.columns]]

    # Save to processed parquet
    output_path = PROCESSED / "interactions.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Interactions data built and saved to {output_path}")

if __name__ == "__main__":
    build_interactions()