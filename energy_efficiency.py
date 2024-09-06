# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

# synthetic data for features; honestly kaggle this at some point
np.random.seed(0)  # reproducibility and predictability 
data_size = 500
data = {
    'Wall_Area': np.random.randint(200, 400, data_size),
    'Roof_Area': np.random.randint(100, 200, data_size),
    'Height': np.random.uniform(3, 10, data_size),
    'Glazing': np.random.uniform(0, 1, data_size),
    'Energy_Efficiency': np.random.uniform(10, 50, data_size)  # Energy efficiency rating
}
df = pd.DataFrame(data)

X = df.drop('Energy_Efficiency', axis=1)
y = df['Energy_Efficiency']

# Plot relationship between Energy Efficiency (target var) & features
sns.pairplot(df, x_vars=['Wall_Area', 'Roof_Area', 'Height', 'Glazing'], y_vars='Energy_Efficiency', height=4, aspect=1, kind='scatter')
plt.show()

# Split synthetic data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80:20 train test

# Train a Random Forest model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Plotting True values vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True Values vs Predicted Values")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.show()
