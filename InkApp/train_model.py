import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import os

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

data = pd.concat(all_dfs, ignore_index=True).dropna()
data['Color'] = data['Color'].map({'Cyan': 0, 'Magenta': 1, 'Yellow': 2, 'Black': 3})
data['Paper type'] = data['Paper type'].map({'Coated': 0, 'Uncoated': 1})
data['Ink key zero setting'] = data['Ink key zero setting'].astype(str).str.replace("mm","",regex=False).astype(float)
data['Delta E improvement'] = data['Delta E before'] - data['Delta E after']

features = ['Color', 'Paper type', 'Zone number', 'Ink key zero setting', 'Delta E improvement', 'initial density', 'initial ink key setting']
X = data[features]
y = data['final ink key setting']

# MEMORY OPTIMIZED: Reduced estimators and depth for 512MB RAM
model = RandomForestRegressor(n_estimators=40, max_depth=7, random_state=42)
model.fit(X, y)

# Save with max compression
joblib.dump(model, MODEL_NAME, compress=9)
print(f"✅ Slim Model Saved. R2 Score: {r2_score(y, model.predict(X)):.4f}")