import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# --- 1. SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = 'consolidated_ink_key_model.pkl'

files = {
    'main': 'Plus real (0.3-11%).xlsx',
    'pakka': 'real job dataset ( pakka wala ).xlsx',
    'five_mm': 'Sample data for 5mm (34_).xlsx',
    'ext1': 'Extended data of plus offset, 2.7mm and 3.9mm (1).xlsx',
    'ext2': 'Extended data of plus offset(11_), 2.7mm(24_) and 3.9mm(33_) and 1.3mm(15_) and 0.6mm(7_) (1).xlsx'
}

all_dfs = []
for key, path in files.items():
    full_path = os.path.join(BASE_DIR, path)
    if os.path.exists(full_path):
        df = pd.read_excel(full_path)
        df.rename(columns={'Paper Type': 'Paper type', 'Delta E after ': 'Delta E after',
                           'Initial Density': 'initial density', 'Initial Ink Key Setting': 'initial ink key setting'}, inplace=True)
        all_dfs.append(df)

# --- 2. CLEAN & FEATURE ENGINEER ---
data = pd.concat(all_dfs, ignore_index=True).dropna()
data['Color'] = data['Color'].map({'Cyan': 0, 'Magenta': 1, 'Yellow': 2, 'Black': 3})
data['Paper type'] = data['Paper type'].map({'Coated': 0, 'Uncoated': 1})
data['Ink key zero setting'] = data['Ink key zero setting'].astype(str).str.replace("mm","",regex=False).astype(float)
data['Delta E improvement'] = data['Delta E before'] - data['Delta E after']

features = ['Color', 'Paper type', 'Zone number', 'Ink key zero setting', 'Delta E improvement', 'initial density', 'initial ink key setting']
X = data[features]
y = data['final ink key setting']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# --- 3. THE SLIM TOURNAMENT ---
# We keep all models, but we limit 'max_depth' and 'n_estimators' to save RAM
models = {
    'Linear': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42),
    'GradientBoost': GradientBoostingRegressor(n_estimators=80, max_depth=4, learning_rate=0.1, random_state=42)
}

best_r2 = -1
best_model = None
best_model_name = ""

print(f"{'Model':<15} | {'R2 Score':<10} | {'MSE':<10}")
print("-" * 40)

for name, model in models.items():
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    print(f"{name:<15} | {r2:<10.4f} | {mse:<10.4f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name

# --- 4. SAVE THE WINNER ---
# compress=9 is critical for staying under the 100MB Git/Render file limit
joblib.dump(best_model, MODEL_NAME, compress=9)
print(f"\n🏆 Winner: {best_model_name} (R2: {best_r2:.4f}) saved as '{MODEL_NAME}'")