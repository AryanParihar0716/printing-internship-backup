"""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the dataset
data = pd.read_excel('./Plus real (0.3-11_).xlsx')
ed3 = pd.read_excel('real job dataset ( pakka wala ).xlsx')
fivemmdata = pd.read_excel('Sample data for 5mm (34_).xlsx')
ed1 = pd.read_excel('Extended data of plus offset, 2.7mm and 3.9mm (1).xlsx')
ed2 = pd.read_excel('Extended data of plus offset(11_), 2.7mm(24_) and 3.9mm(33_) and 1.3mm(15_) and 0.6mm(7_) (1).xlsx')
data = data.dropna()  # Drop rows with missing values
data['Color']= data['Color'].map({'Cyan': 0, 'Magenta': 1, 'Yellow': 2, 'Black': 3})
data['Paper type'] = data['Paper type'].map({'Coated': 0, 'Uncoated': 1})
data['Ink key zero setting'] = data['Ink key zero setting'].str.replace("mm","",regex=False).astype(float)
data['Delta E improvement'] = data['Delta E before'] - data['Delta E after']
X = data[['Color', 'Paper type','Zone number','Ink key zero setting','Delta E improvement', 'initial density','initial ink key setting']]
y = data['final ink key setting']
# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# Create a linear regression model
model = LinearRegression()
# Train the model
model.fit(x_train, y_train)
# Make predictions
y_pred = model.predict(x_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Linear Regression - Mean Squared Error: {mse}")
print(f"Linear Regression - R^2 Score: {r2}")

# Create a decision tree regression model
dt_model = DecisionTreeRegressor(random_state=42)
# Train the model
dt_model.fit(x_train, y_train)
# Make predictions
y_pred_dt = dt_model.predict(x_test)
# Evaluate the model
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
print(f"Decision Tree - Mean Squared Error: {mse_dt}")
print(f"Decision Tree - R^2 Score: {r2_dt}")

# Create a random forest regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# Train the model
rf_model.fit(x_train, y_train)
# Make predictions
y_pred_rf = rf_model.predict(x_test)
# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest - Mean Squared Error: {mse_rf}")
print(f"Random Forest - R^2 Score: {r2_rf}")

# Create a gradient boosting regression model
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
# Train the model
gb_model.fit(x_train, y_train)
# Make predictions
y_pred_gb = gb_model.predict(x_test)
# Evaluate the model
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)
print(f"Gradient Boosting - Mean Squared Error: {mse_gb}")
print(f"Gradient Boosting - R^2 Score: {r2_gb}")

new_data = pd.DataFrame({'Linear Output': y_pred, 'Decision Tree Output': y_pred_dt, 'Random Forest Output': y_pred_rf, 'Gradient Boosting Output': y_pred_gb})
new_data = pd.concat([pd.Series(y_test.values, name='Actual Ink Key Setting'), new_data], axis=1)
print(new_data.head())
new_data.to_excel('model_predictions.xlsx', index=False)"""

"""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump

# 1. Load all datasets
df_main = pd.read_excel('Plus real (0.3-11%).xlsx')
ed3 = pd.read_excel('real job dataset ( pakka wala ).xlsx')
fivemmdata = pd.read_excel('Sample data for 5mm (34_).xlsx')
ed1 = pd.read_excel('Extended data of plus offset, 2.7mm and 3.9mm (1).xlsx')
ed2 = pd.read_excel('Extended data of plus offset(11_), 2.7mm(24_) and 3.9mm(33_) and 1.3mm(15_) and 0.6mm(7_) (1).xlsx')

# 2. Harmonize column names before concatenation
# (Ensuring standard names across all files)
for df in [fivemmdata, ed1, ed2]:
    df.rename(columns={
        'Paper Type': 'Paper type', 
        'Delta E after ': 'Delta E after'
    }, inplace=True)

# 3. Combine into one master dataframe
data = pd.concat([df_main, ed3, fivemmdata, ed1, ed2], ignore_index=True)
data = data.dropna()

# 4. Feature Engineering & Cleaning
data['Color'] = data['Color'].map({'Cyan': 0, 'Magenta': 1, 'Yellow': 2, 'Black': 3})
data['Paper type'] = data['Paper type'].map({'Coated': 0, 'Uncoated': 1})
data['Ink key zero setting'] = data['Ink key zero setting'].astype(str).str.replace("mm","",regex=False).astype(float)
data['Delta E improvement'] = data['Delta E before'] - data['Delta E after']

# 5. Define X and y
X = data[['Color', 'Paper type', 'Zone number', 'Ink key zero setting', 'Delta E improvement', 'initial density', 'initial ink key setting']]
y = data['final ink key setting']

# 6. Split Dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# --- 7. Model Training & Evaluation ---

results = {}

# Linear Regression
model_lr = LinearRegression().fit(x_train, y_train)
y_pred_lr = model_lr.predict(x_test)
results['Linear'] = y_pred_lr

# Decision Tree
model_dt = DecisionTreeRegressor(random_state=42).fit(x_train, y_train)
y_pred_dt = model_dt.predict(x_test)
results['DT'] = y_pred_dt

# Random Forest
model_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(x_train, y_train)
y_pred_rf = model_rf.predict(x_test)
results['RF'] = y_pred_rf

# Gradient Boosting
model_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(x_train, y_train)
y_pred_gb = model_gb.predict(x_test)
results['GB'] = y_pred_gb

# 8. Print Results
print("--- Comparison of Consolidated Models ---")
for name in results:
    r2 = r2_score(y_test, results[name])
    mse = mean_squared_error(y_test, results[name])
    print(f"{name:18} | R2: {r2:.4f} | MSE: {mse:.4f}")

# 9. Save all predictions to Excel for comparison
comparison_df = pd.DataFrame({
    'Actual': y_test.values,
    'Linear': results['Linear'],
    'Decision Tree': results['DT'],
    'Random Forest': results['RF'],
    'Gradient Boosting': results['GB']
})
comparison_df.to_excel('consolidated_model_test.xlsx', index=False)
print("\n✅ Combined testing complete. Results saved to 'consolidated_model_test.xlsx'")"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Load all 5 datasets
files = {
    'main': 'Plus real (0.3-11%).xlsx',
    'pakka': 'real job dataset ( pakka wala ).xlsx',
    'five_mm': 'Sample data for 5mm (34_).xlsx',
    'ext1': 'Extended data of plus offset, 2.7mm and 3.9mm (1).xlsx',
    'ext2': 'Extended data of plus offset(11_), 2.7mm(24_) and 3.9mm(33_) and 1.3mm(15_) and 0.6mm(7_) (1).xlsx'
}

all_dfs = []
for key, path in files.items():
    df = pd.read_excel(path)
    # Standardize Column Names across different file versions
    df.rename(columns={
        'Paper Type': 'Paper type', 
        'Delta E after ': 'Delta E after',
        'Initial Density': 'initial density',
        'Initial Ink Key Setting': 'initial ink key setting'
    }, inplace=True)
    all_dfs.append(df)

# 2. Combine and Clean
data = pd.concat(all_dfs, ignore_index=True).dropna()

# 3. Feature Engineering
data['Color'] = data['Color'].map({'Cyan': 0, 'Magenta': 1, 'Yellow': 2, 'Black': 3})
data['Paper type'] = data['Paper type'].map({'Coated': 0, 'Uncoated': 1})
data['Ink key zero setting'] = data['Ink key zero setting'].astype(str).str.replace("mm","",regex=False).astype(float)
data['Delta E improvement'] = data['Delta E before'] - data['Delta E after']

# 4. Feature Selection
features = ['Color', 'Paper type', 'Zone number', 'Ink key zero setting', 'Delta E improvement', 'initial density', 'initial ink key setting']
X = data[features]
y = data['final ink key setting']

# 5. Split Dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 6. Train and Compare
models = {
    'Linear': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoost': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

best_r2 = -1
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
        best_model_name = name

# 7. Save the absolute Best Model for the App
joblib.dump(models[best_model_name], 'consolidated_ink_key_model.pkl')
print(f"\n✅ Winner: {best_model_name} saved as 'consolidated_ink_key_model.pkl'")