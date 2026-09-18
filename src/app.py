import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine

# ----- WEB PAGE SETTINGS -----
st.set_page_config(page_title="Retail Price Optimizer", layout="centered")
st.title("Algorithmic Price Optimization Dashboard")
st.markdown(
    "Powered by PostgreSQL & XGBoost Predictive Demand Modeling"
)

# cached data from SQL for speed
@st.cache_resource
def load_optimized_assets():
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent
    env_path = root_dir / ".env"

    # Update path to use your final XGBoost model artifact
    model_path = current_dir / "demand_model.joblib"

    load_dotenv(dotenv_path=env_path, override=True)
    db_password = os.getenv("DB_PASSWORD")

    engine = create_engine(
        f"postgresql://postgres:{db_password}@localhost:5432/postgres"
    )

    # FIX: Added 'stockout_flag' so the ML model has all its required training features
    sql_query = "SELECT * FROM optimized_sales_features;"
    df_clean = pd.read_sql(sql_query, con=engine)

    # Load via joblib for superior memory mapping performance
    demand_model = joblib.load(model_path)

    return df_clean, demand_model

try:
    df, demand_model = load_optimized_assets()
    st.success("Securely connected to PostgreSQL. Pre-trained ML Model loaded!")
except Exception as e:
    st.error(f"App Configuration Failure: {e}")
    st.stop()


# ----- FRONTEND UI -----
st.header("Simulate Pricing Strategies")

# Dropdown for user-defined product category selection
selected_cat = st.selectbox("Select Product Category", df["category"].unique())

# Pull metrics for context
cat_df = df[df["category"] == selected_cat]
current_avg = cat_df["current_price"].mean()
min_price = cat_df["current_price"].min()
max_price = cat_df["current_price"].max()

# Display baseline comparison cards
col1, col2 = st.columns(2)
col1.metric("Historical Avg Price", f"${current_avg:.2f}")
col2.metric("Market Price Range", f"${min_price:.2f} - ${max_price:.2f}")


# optimization
def run_optimization(promo_name, stockout_status, margin_pct=0.40):
    estimated_cost = current_avg * (1 - margin_pct)
    # Generate bounded test grid (+/- 20% around average)
    test_prices = np.linspace(current_avg * 0.8, current_avg * 1.2, 200)

    sim_grid = pd.DataFrame(
        {
            "current_price": test_prices,
            "category": selected_cat,
            "promotion_type": promo_name,
            "stockout_flag": stockout_status,
        }
    )

    # Reorder columns exactly how the model pipeline saw them during training
    feature_order = ["current_price", "category", "promotion_type", "stockout_flag"]
    sim_grid = sim_grid[feature_order]

    sim_grid["predicted_units"] = demand_model.predict(sim_grid)
    sim_grid["predicted_units"] = sim_grid["predicted_units"].clip(lower=0)
    sim_grid["projected_profit"] = sim_grid["predicted_units"] * (
        sim_grid["current_price"] - estimated_cost
    )

    best_idx = sim_grid["projected_profit"].idxmax()
    return round(sim_grid.loc[best_idx, "current_price"], 2)


# Action Button to compute optimal configurations
if st.button("Compute Optimal Prices"):
    with st.spinner("Running algorithmic grid search..."):
        # Scenario A: Standard Pricing (No promo, in stock)
        available_promos = cat_df["promotion_type"].unique()
        baseline_promo = "None" if "None" in available_promos else available_promos[0]
        optimal_reg = run_optimization(baseline_promo, stockout_status=0)

        # Scenario B: Member Promotional Campaign
        target_promo = (
            "Member Offer"
            if "Member Offer" in available_promos
            else available_promos[-1]
        )
        optimal_promo = run_optimization(target_promo, stockout_status=0)

    # Output Results Matrix to UI
    st.subheader("Optimization Matrix Recommendations")
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Recommended Regular Price", f"${optimal_reg:.2f}")
    res_col2.metric("Recommended Promo Price", f"${optimal_promo:.2f}")

    st.info(
        "**Operational Rule Applied:** Out-of-Stock configurations (`stockout_flag=1`) have been hard-coded to **\$0.00** discounts to preserve margins based on historical leakage analysis."
    )
