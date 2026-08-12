# 🚗 Car Price Prediction Engine

An end-to-end Machine Learning pipeline and interactive web UI for predicting used car prices based on vehicle specifications.

---

## 🌟 Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 📌 Project Architecture & Features

1. **Exploratory Data Analysis (EDA):**
   * Target variable (`Price`) distribution analysis.
   * Feature correlation heatmaps and multicollinearity check.
   * Categorical feature impact analysis.

2. **Advanced Preprocessing & Pipeline:**
   * Custom cleaning for dataset-specific formats (`Levy`, `Mileage`, `Engine Volume`, `Turbo`).
   * Outlier mitigation using the Interquartile Range (IQR) method.
   * Scikit-Learn `Pipeline` integration with `ColumnTransformer`:
     * Numeric Scaling (`RobustScaler`).
     * Categorical Encoding (`OneHotEncoder`).
     * Dimensionality Reduction using PCA for high-cardinality features (`Manufacturer`, `Model`).

3. **Machine Learning Models:**
   * Trained and evaluated baseline and ensemble models: **Random Forest**, **XGBoost**, and **LightGBM**.
   * Selected the optimal model based on **RMSE**, **MAE**, **R²**, and **Adjusted R²**.

4. **Web Deployment:**
   * Interactive Streamlit application for real-time predictions.

---

## 📂 Repository Structure

```text
├── app.py                      # Streamlit UI & Inference Logic
├── train_pipeline.py           # ML Model Training & Preprocessing Pipeline
├── car_price_prediction.ipynb  # EDA & Analysis Notebook (Colab)
├── car_price_model.pkl         # Serialized Trained Model Pipeline
├── car_price_prediction.csv    # Dataset
├── requirements.txt            # Python Dependencies
├── correlation_matrix.png      # Feature Correlation Plot
└── price_distribution.png     # Price Distribution Plot


🚀 Quick Start (Local Setup)
1-Clone the repository:

Bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)
cd YOUR_REPOSITORY


2-Install dependencies:

Bash
pip install -r requirements.txt


3-Train the model (Optional):

Bash
python train_pipeline.py


4-Run the Streamlit application:

Bash
streamlit run app.py