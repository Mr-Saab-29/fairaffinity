from __future__ import annotations
from src.features.product_features import build_product_features
from src.features.user_features import build_user_features
from src.features.rfm_features import build_user_rfm, build_client_product_recency
from src.features.category_features import build_category_features

def build_all_features(cutoff: str | None = None) -> None:
    """Build all features for the project.

    Args:
        cutoff (str | None, optional): Cutoff date for feature calculations. Defaults to None.
    """
    print("Building user features...")
    build_user_features(cutoff)
    
    print("Building product features...")
    build_product_features(cutoff)

    print("Building RFM Features...")
    build_user_rfm(cutoff)

    print("Building Client Product Recency Features...")
    build_client_product_recency(cutoff)

    print("Building Category Features...")
    build_category_features(cutoff)

    print("All features built successfully!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build all features for the project.")
    parser.add_argument("--cutoff", type=str, help="Cutoff date in YYYY-MM-DD format. Defaults to None.", default=None)
    
    args = parser.parse_args()
    
    build_all_features(args.cutoff)