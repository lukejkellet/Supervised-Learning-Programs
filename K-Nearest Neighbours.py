import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

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

#Initialisation and fitting
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)

#Evaluating the performance of the model using R2 score and mean squared error
prediction = model.predict(X_test)

r2 = r2_score(y_test, prediction)
mse = mean_squared_error(y_test, prediction)

print("Mean squared error: ", mse)
print("R2 score: ", r2)

#Feature Weight Plotting
perm_weight = permutation_importance(model, X_test, y_test)
sorted_idx = perm_weight.importances_mean.argsort()
sorted_weight = perm_weight.importances_mean[sorted_idx]
sorted_features = np.array(X.columns.tolist())[sorted_idx]

# Plotting feature weights as horizontal chart in ascending order
plt.barh(range(len(sorted_weight)), sorted_weight)
plt.yticks(range(len(sorted_weight)), sorted_features)
plt.xlabel('Importance')
plt.title("Feature Weights - K-Nearest Neighbours Regressor")
plt.show()
