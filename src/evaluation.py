import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def optimize_price_via_grid(category_name, promo_type, is_stockout, clean_df, demand_model, margin=0.40):
    # grid search price
    cat_data = clean_df[clean_df["category"] == category_name]

    if cat_data.empty:
        return 0.0

    min_market_price = cat_data["current_price"].min()
    max_market_price = cat_data["current_price"].max()
    avg_market_price = cat_data["current_price"].mean()

    # estimated unit production cost based on margin parameters
    estimated_cost = avg_market_price * (1 - margin)

    # restrict price search (+/- 20% of the current average)
    potential_prices = np.linspace(avg_market_price * 0.8, avg_market_price * 1.2, 300)

    # simulation matrix
    simulation_grid = pd.DataFrame(
        {
            "current_price": potential_prices,
            "category": category_name,
            "promotion_type": promo_type,
            "stockout_flag": is_stockout,
        }
    )

    # feed the simulation grid directly to hyper-tuned model
    simulation_grid["predicted_units_sold"] = demand_model.predict(
        simulation_grid
    )
    simulation_grid["predicted_units_sold"] = simulation_grid[
        "predicted_units_sold"
    ].clip(lower=0)

    # Profit Equation = Units Sold * (Price - Cost)
    simulation_grid["projected_profit"] = simulation_grid[
        "predicted_units_sold"
    ] * (simulation_grid["current_price"] - estimated_cost)

    # grab the price point that returned the highest profit metric
    best_row_idx = simulation_grid["projected_profit"].idxmax()
    optimal_price = simulation_grid.loc[best_row_idx, "current_price"]

    return round(optimal_price, 2)


def run_pipeline(df):
    print("----- INITIALIZING ENGINE -----\n")

    # ----- TRAIN/TEST SPLIT -----
    # selecting features
    features = ["current_price", "category", "promotion_type", "stockout_flag"]

    clean_df = df.dropna(subset=features + ["units_sold", "revenue"]).copy()

    X = clean_df[features]
    y = clean_df["units_sold"]

    # 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(
        f"Dataset split: {len(X_train):,} training rows | {len(X_test):,} validation rows."
    )


    # ----- PREPROCESSING -----
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["current_price"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                ["category", "promotion_type", "stockout_flag"],
            ),
        ]
    )


    # ----- HYPERPARAM TUNING -----
    base_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(random_state=42)),
        ]
    )

    # define grid
    param_grid = {
        "regressor__n_estimators": [50, 100, 200],
        "regressor__max_depth": [10, 20, None],
    }

    print("Running GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,  
    )

    grid_search.fit(X_train, y_train)

    # extract the best model
    demand_model = grid_search.best_estimator_
    print(f"Grid Search complete! Optimal Parameters: {grid_search.best_params_}")


    # ----- METRICS EVALUATION -----
    # historical average baseline
    baseline_guesses = [y_train.mean()] * len(y_test)
    baseline_mae = mean_absolute_error(y_test, baseline_guesses)

    # hyper-tuned model
    ml_predictions = demand_model.predict(X_test)
    ml_mae = mean_absolute_error(y_test, ml_predictions)

    # error reduction formula
    error_reduction_pct = ((baseline_mae - ml_mae) / baseline_mae) * 100

    print("----------------------------------------------------")
    print("PERFORMANCE METRICS REPORT")
    print("----------------------------------------------------")
    print(f"Historical Baseline Average MAE: {baseline_mae:.2f} units")
    print(f"Hyper-Tuned Random Forest MAE:   {ml_mae:.2f} units")
    print("----------------------------------------------------")
    print(f"TOTAL ERROR REDUCTION (MAE):     {error_reduction_pct:.2f}%")
    print("----------------------------------------------------")
    return demand_model, clean_df


def price_optimization(clean_df, demand_model):
    # ----- PRICE MANAGEMENT -----
    optimization_records = []
    categories = clean_df["category"].unique()

    print("Simulating algorithmic grid search price points...")
    for cat in categories:
        cat_df = clean_df[clean_df["category"] == cat]
        current_avg = cat_df["current_price"].mean()

        # identify available promos
        available_promos = cat_df["promotion_type"].unique()
        baseline_promo = "None" if "None" in available_promos else available_promos[0]
        target_promo = (
            "Member Offer"
            if "Member Offer" in available_promos
            else available_promos[-1]
        )

        # run simulations
        opt_regular_price = optimize_price_via_grid(
            cat, baseline_promo, 0, clean_df, demand_model
        )
        opt_promo_price = optimize_price_via_grid(
            cat, target_promo, 0, clean_df, demand_model
        )

        optimization_records.append(
            {
                "Category": cat,
                "Current Avg Price": round(current_avg, 2),
                "Optimal Regular Price": opt_regular_price,
                "Optimal Promo Price": opt_promo_price,
            }
        )

    optimized_price_grid = pd.DataFrame(optimization_records)
    print("\n--------------------------------------------------")
    print("PRICING GRID RESULT")
    print("----------------------------------------------------")
    return optimized_price_grid


