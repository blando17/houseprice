# House Price Prediction Using XGBoost

This project demonstrates a machine learning approach to predict house prices using the California Housing dataset and the XGBoost Regressor model.

## 📂 Project Structure
- Dataset: California Housing Dataset (loaded using sklearn)
- Model: XGBoost Regressor
- Environment: Google Colab / Jupyter Notebook

##  Key Steps

### 1. Importing Required Libraries
- `numpy`, `pandas` for data handling
- `matplotlib.pyplot`, `seaborn` for visualization
- `sklearn.datasets`, `sklearn.model_selection`, `sklearn.metrics`
- `xgboost.XGBRegressor` for model training

### 2. Loading Dataset
- California Housing dataset is fetched using `fetch_california_housing()` from `sklearn.datasets`.

### 3. Data Preparation
- Converted to a DataFrame and added a new column `price` (target).
- Shape: (20640, 9) — no missing values.
- Performed basic statistical summary and correlation analysis.

### 4. Exploratory Data Analysis
- Generated correlation heatmap to identify feature relationships.

### 5. Data Splitting
- Features (X): All columns except `price`
- Target (Y): `price`
- Data split: 80% training, 20% testing

### 6. Model Training
- Trained `XGBRegressor` on the training dataset using default hyperparameters.

### 7. Model Evaluation
- Metrics:
  - **Training R² score**: 0.94
  - **Training MAE**: 0.19
  - **Testing R² score**: 0.83
  - **Testing MAE**: 0.31
- Visualized Actual vs Predicted prices using a scatter plot.


## 📊 Output
The model demonstrates good predictive performance and generalization with a reasonable error margin on test data.

