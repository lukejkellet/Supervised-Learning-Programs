import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

#Loading the dataset
#data = pd.read_csv("./Datasets/Dataset1/exams.csv") #First dataset
#data = pd.read_csv("./Datasets/Dataset2/heart.csv") #Second dataset
data = pd.read_csv("./Datasets/Dataset3/train.csv") #Third dataset

#target = 'passed' #First dataset
#target = 'output' #Second dataset
target = 'price_range' #Third dataset

# Preprocessing Techniques (One-hot encoding and Feature Scaling)
cat_cols = data.select_dtypes(include=['object', 'bool']).columns.tolist()
data = pd.get_dummies(data, columns=cat_cols)

X = data.drop(target, axis=1) #Dropping the target column
y = data[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test/Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialisation and fitting
model = LinearRegression()
model.fit(X_train, y_train)

#Evaluating the performance of the model using MSE and R2
prediction = model.predict(X_test)

mse = mean_squared_error(y_test, prediction)
r2 = r2_score(y_test, prediction)

print("Mean Squared Error (MSE): ", mse)
print("R-squared (R2): ", r2)

#Coefficient Plotting
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
coefficients = coefficients.sort_values('Coefficient')
plt.barh(coefficients['Feature'], coefficients['Coefficient'])
plt.title("Feature Weights - Linear Regression")
plt.show()
