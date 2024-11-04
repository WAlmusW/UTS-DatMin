import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the CSV file
df = pd.read_csv('Linear Regression/delivery_data.csv')

# Define the independent variables (JP, JT, J) and the dependent variable (T)
X = df[['JP', 'JT', 'J']].values  # Independent variables
Y = df['T'].values                # Dependent variable

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, Y)

# Extract coefficients
intercept = model.intercept_
coefficients = model.coef_

print(f"Linear Regression Model: T = {intercept} + {coefficients[0]}*JP + {coefficients[1]}*JT + {coefficients[2]}*J")

# Predict for the new data point: JP=4, JT=2, J=5
new_data = [[4, 2, 5]]
predicted_T = model.predict(new_data)[0]
print(f"Predicted T for JP=4, JT=2, J=5 is {predicted_T}")
