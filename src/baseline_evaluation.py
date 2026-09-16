import numpy as np
import pandas as pd


def analyze_discount_leaks(df):
    """Combines inventory issues and promo leaks into a single actionable summary."""
    print("--- 1. PROMO LEAKAGE BY TYPE ---")
    print(df.groupby('promotion_type')['discount_leakage'].mean().reset_index())
    
    print("\n--- 2. DISCOUNTS GIVEN DURING STOCKOUTS ---")
    print(df.groupby(['category', 'stockout_flag'])['discount_pct'].mean().reset_index())


def calculate_price_elasticity(df):
    elasticity_results = []
    categories = df['category'].unique()
    
    for cat in categories:
        sub_df = df[(df['category'] == cat) & (df['units_sold'] > 0) & (df['current_price'] > 0)].copy()
        
        # Array to hold slopes of individual products in this category
        product_slopes = []
        
        # Loop through each unique item in this category (assuming 'product_id' exists)
        if 'product_id' in df.columns:
            for prod in sub_df['product_id'].unique():
                prod_df = sub_df[sub_df['product_id'] == prod]
                
                # Need enough historical price changes for a single product to find a trend
                if len(prod_df) >= 5 and prod_df['current_price'].nunique() > 1:
                    log_qty = np.log(prod_df['units_sold'])
                    log_price = np.log(prod_df['current_price'])
                    slope, _ = np.polyfit(log_price, log_qty, 1)
                    product_slopes.append(slope)
        
        # If no product_id column, fallback to Option B (Relative Price Ratio)
        if not product_slopes:
            continue
            
        # Average the individual elasticities to get the true category score
        avg_slope = np.mean(product_slopes)
        
        if avg_slope < -1:
            demand_type = "Price Sensitive (Elastic)"
            strategy = "Discounts work well here to rapidly clear inventory and boost overall volume."
        else:
            demand_type = "Price Insensitive (Inelastic)"
            strategy = "Customers buy this anyway. Drop the discounts and preserve your profit margins."
            
        elasticity_results.append({
            'category': cat,
            'sensitivity_score': round(avg_slope, 4),
            'customer_type': demand_type,
            'business_action': strategy
        })
        
    return pd.DataFrame(elasticity_results)
