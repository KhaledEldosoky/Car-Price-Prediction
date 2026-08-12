import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Scikit-Learn & Modeling
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ==========================================
# PHASE 1: Exploratory Data Analysis (EDA) & Cleaning
# ==========================================
def load_and_clean_data(filepath):
    print("Loading data...")
    df = pd.read_csv(filepath)

    # Drop irrelevant ID column
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)

    print("Initial Data Cleaning...")
    # 1. Clean Levy (Replace '-' with NaN, convert to float)
    df['Levy'] = df['Levy'].replace('-', np.nan).astype(float)

    # 2. Clean Mileage (Remove ' km' and convert to float)
    df['Mileage'] = df['Mileage'].astype(str).str.replace(' km', '').astype(float)

    # 3. Clean Engine volume (Extract numeric, create a Turbo indicator)
    df['Turbo'] = df['Engine volume'].astype(str).str.contains('Turbo').astype(int)
    df['Engine volume'] = df['Engine volume'].astype(str).str.replace(' Turbo', '').astype(float)

    # 4. Clean Doors (Standardize weird date-like strings)
    door_mapping = {'04-May': '4-5', '02-Mar': '2-3', '>5': '>5'}
    df['Doors'] = df['Doors'].map(door_mapping)

    # Remove extreme outliers in Price (e.g., typos in price, $1 cars, etc.)
    df = df[(df['Price'] > 500) & (df['Price'] < 500000)]

    return df

def perform_eda(df):
    print("Performing EDA...")
    # 1. Correlation Matrix Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.savefig('correlation_matrix.png')
    plt.close()

    # 2. Target Variable Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Price'], bins=50, kde=True)
    plt.title("Price Distribution")
    plt.savefig('price_distribution.png')
    plt.close()

# ==========================================
# PHASE 2: Advanced Preprocessing & Engineering
# ==========================================
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def build_preprocessing_pipeline(num_features, cat_features_low_card, cat_features_high_card):
    # Numeric Pipeline: Impute -> RobustScaler (handles remaining outliers)
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    # Low Cardinality Categorical Pipeline: Impute -> OneHotEncode
    cat_low_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # High Cardinality Categorical Pipeline: Impute -> OneHotEncode -> PCA
    # PCA reduces the massive dimensionality of Models/Manufacturers
    cat_high_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('pca', PCA(n_components=50, random_state=42))
    ])

    # Combine via ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_features),
        ('cat_low', cat_low_pipeline, cat_features_low_card),
        ('cat_high', cat_high_pipeline, cat_features_high_card)
    ])

    return preprocessor

# ==========================================
# PHASE 3 & 4: Model Training, Eval & Comparison
# ==========================================
def evaluate_model(y_true, y_pred, n, p):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    return rmse, mae, r2, adj_r2

def main():
    # 1. Load Data
    df = load_and_clean_data("car_price_prediction.csv")

    # IQR Outlier mitigation on target variable to stabilize training
    df = remove_outliers_iqr(df, 'Price')

    perform_eda(df)

    # Feature Segregation
    target = 'Price'
    X = df.drop(target, axis=1)
    y = df[target]

    num_features = ['Levy', 'Prod. year', 'Engine volume', 'Mileage', 'Cylinders', 'Airbags', 'Turbo']
    cat_low_card = ['Category', 'Leather interior', 'Fuel type', 'Gear box type', 'Drive wheels', 'Doors', 'Wheel']
    cat_high_card = ['Manufacturer', 'Model', 'Color']

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Build Preprocessor
    preprocessor = build_preprocessing_pipeline(num_features, cat_low_card, cat_high_card)

    # 4. Define Models
    models = {
        'RandomForest (Baseline)': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
        'LightGBM': LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    }

    results = []
    best_model = None
    best_r2 = -float('inf')
    best_pipeline = None

    n = len(X_test)
    p = X_train.shape[1]

    print("Training Models...")
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)

        # Evaluate
        rmse, mae, r2, adj_r2 = evaluate_model(y_test, y_pred, n, p)
        results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R-Squared": r2, "Adj R-Squared": adj_r2})

        if r2 > best_r2:
            best_r2 = r2
            best_model = name
            best_pipeline = pipeline

    # 5. Display Comparison Table
    results_df = pd.DataFrame(results).sort_values(by="RMSE")
    print("\n--- Model Evaluation Comparison ---")
    print(results_df.to_string(index=False))

    print(f"\n🏆 Best Model: {best_model} with R2: {best_r2:.4f}")

    # 6. Save the Winning Model Pipeline
    joblib.dump(best_pipeline, 'car_price_model.pkl')
    print("Winning pipeline saved to 'car_price_model.pkl'")

if __name__ == "__main__":
    main()