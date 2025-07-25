import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("Calories1.csv")

# Preview
print("First 5 rows:\n", df.head())
print("\nShape of the dataset:", df.shape)
print("\nInfo:\n")
df.info()
print("\nDescription:\n", df.describe())

# Scatter plot: Height vs Weight
sb.scatterplot(x='Height', y='Weight', data=df)
plt.title("Height vs Weight")
plt.show()

# Scatter plots for key features vs Calories
features_to_plot = ['Age', 'Height', 'Weight', 'Duration']
plt.subplots(figsize=(15, 10))
for i, col in enumerate(features_to_plot):
    plt.subplot(2, 2, i + 1)
    sample = df.sample(1000)
    sb.scatterplot(x=col, y='Calories', data=sample)
    plt.title(f'{col} vs Calories')
plt.tight_layout()
plt.show()

# Distribution plots for float columns
float_features = df.select_dtypes(include='float').columns
plt.subplots(figsize=(15, 10))
for i, col in enumerate(float_features):
    plt.subplot(2, 3, i + 1)
    sb.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Encode Gender: male=0, female=1
df['Gender'].replace({'male': 0, 'female': 1}, inplace=True)

# Heatmap for high correlation
plt.figure(figsize=(8, 8))
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.title("Highly Correlated Feature Pairs")
plt.show()

# Drop highly correlated features
df.drop(['Weight', 'Duration'], axis=1, inplace=True)

# Define features and target
X = df.drop(['User_ID', 'Calories'], axis=1)
y = df['Calories'].values

# Train-Test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=22)

print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

models = [
    ("Linear Regression", LinearRegression()),
    ("XGBoost Regressor", XGBRegressor(verbosity=0)),
    ("Lasso Regression", Lasso()),
    ("Random Forest Regressor", RandomForestRegressor()),
    ("Ridge Regression", Ridge())
]

print("\nModel Evaluation:\n")
for name, model in models:
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    print(f'{name}:')
    print('  Training MAE:', mean_absolute_error(y_train, y_train_pred))
    print('  Validation MAE:', mean_absolute_error(y_val, y_val_pred))
    print('  Validation R² Score:', r2_score(y_val, y_val_pred))
    print('-' * 50)

print("Prediction Completed.!")