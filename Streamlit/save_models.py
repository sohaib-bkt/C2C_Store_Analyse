import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
import pickle

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Load dataset
sellers = pd.read_csv('data/seller.csv')

# Define features and target
features = ['totalproductslisted', 'totalbought', 'totalwished', 'totalproductsliked']
target = 'totalproductssold'

X = sellers[features]
y = sellers[target]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Ridge and Lasso models
ridge = Ridge(alpha=1)
ridge.fit(X_train, y_train)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Save models and scaler
with open('models/ridge_model.pkl', 'wb') as f:
    pickle.dump(ridge, f)

with open('models/lasso_model.pkl', 'wb') as f:
    pickle.dump(lasso, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Models and scaler saved successfully!")
