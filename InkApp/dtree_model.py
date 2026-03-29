import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the dataset
data = pd.read_excel('Plus real (0.3-11%).xlsx')
pdata= pd.read_excel('real job dataset ( pakka wala ).xlsx')
fivemmdata = pd.read_excel('Sample data for 5mm (34_).xlsx')
edata1 = pd.read_excel('Extended data of plus offset(11_), 2.7mm(24_) and 3.9mm(33_) and 1.3mm(15_) and 0.6mm(7_) (1).xlsx')
edata2 = pd.read_excel('Extended data of plus offset, 2.7mm and 3.9mm (1).xlsx')
data = pd.concat([data, pdata, fivemmdata, edata1, edata2], ignore_index=True)
data = data.dropna()  # Drop rows with missing values
data['Color']= data['Color'].map({'Cyan': 0, 'Magenta': 1, 'Yellow': 2, 'Black': 3})
data['Paper type'] = data['Paper type'].map({'Coated': 0, 'Uncoated': 1})
data['Ink key zero setting'] = data['Ink key zero setting'].str.replace("mm","",regex=False).astype(float)
data['Delta E improvement'] = data['Delta E before'] - data['Delta E after']
X = data[['Color', 'Paper type','Ink key zero setting','Delta E improvement', 'initial density','initial ink key setting']]
y = data['final ink key setting']
# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# Create a linear regression model


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
prediction =  dt_model.predict(np.array([[3,0,0,(5.77-1.62),1.31,48.03]]))
print(prediction[0])
prediction = dt_model.predict(np.array([[3,0,0,(5.77-1.62),1.31,33.03]]))
print(prediction[0])

import joblib
joblib.dump(dt_model, 'decision_tree_model.pkl')