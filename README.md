# **Demand Forecaster & Price Optimizer**

**An end-to-end machine learning pipeline and business simulation engine that forecasts product demand and algorithmically optimizes pricing for maximum profitability.**

---

## Data Source
* **[Kaggle](https://www.kaggle.com/datasets/noopurbhatt/retail-pricing-and-demand-signals-dataset/data?select=retail_pricing_demand_100k.csv)**: Processed over **170,000 transaction records** capturing granular retail vectors including historical volume metrics, active category layers, cross-channel promotional distributions, and real-time inventory stockout events.


## Technologies Used
* **Python (Pandas, Numpy)**
* **Scikit-learn**
* **XGBoost**
* **Jupyter Notebook**


## Key Features
* Built a data preprocessing pipeline with feature scaling and categorical encoding.
* Created a simulator that tests 300 price permutations per category under different promos.
* Generated an automated table of the exact prices that yield maximum profit.


## Results
* Significantly reduced overall predictive error (MAE) compared to historical averages by 58%.
* Produced an actionable master grid showing current average prices versus optimal profit-maximizing prices.


## Usage Guide
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/vphngt/price-optimization.git
cd price-optimization
pip install -r requirements.txt
```
