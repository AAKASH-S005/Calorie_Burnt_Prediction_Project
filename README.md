# 🔥 Calories Burned Prediction using Machine Learning

This project applies various machine learning regression models to predict the number of calories burned based on physical activity attributes like age, height, weight, duration, etc.

---

## 📊 Project Objective

To build a predictive model that estimates **calories burned** during exercise using physiological features and workout characteristics. This can help in personalized fitness planning and health monitoring.

---

## 🧠 Models Used

The following regression models were trained and compared:

- Linear Regression
- Lasso Regression
- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor

Each model's **Mean Absolute Error (MAE)** and **R² Score** were evaluated for training and validation datasets.

---

## 📁 Dataset

- **File Used:** `Calories1.csv`
- **Attributes Include:**
  - User ID
  - Gender
  - Age
  - Height
  - Weight
  - Duration (in minutes)
  - Heart Rate
  - Body Temperature
  - Calories (Target)

---

## 📌 Key Steps in the Pipeline

1. **Data Loading** and **Exploration**
2. **Visualization**
   - Height vs Weight Scatter Plot
   - Feature vs Calories Scatter Plots
   - Distribution Plots
   - Correlation Heatmap
3. **Preprocessing**
   - Encoding `Gender` (male: 0, female: 1)
   - Dropping highly correlated features
   - Standardizing features
4. **Model Training & Evaluation**
   - Split data (90% train / 10% validation)
   - Train and test all five models
   - Display MAE and R² for comparison

---

## 📦 Dependencies

Install all dependencies with:

```bash
pip install -r requirements.txt

---

### ✅ `requirements.txt`

```txt
numpy==1.24.4
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
xgboost==1.7.6
