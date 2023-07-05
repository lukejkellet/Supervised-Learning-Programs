import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Loading the dataset
#data = pd.read_csv("./Datasets/Dataset1/exams.csv") #First dataset
#data = pd.read_csv("./Datasets/Dataset2/heart.csv") #Second dataset
data = pd.read_csv("./Datasets/Dataset3/train.csv") #Third dataset

#target = 'passed' #First dataset
#target = 'output' #Second dataset
target = 'price_range' #Third dataset

# Preprocessing Techniques (One-hot encoding and Feature Scaling)
cat_cols = data.select_dtypes(include=['object', 'bool']).columns.tolist()
data = pd.get_dummies(data, columns=cat_cols)

num_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
scaler = StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

X = data.drop(target, axis=1) #Dropping the target column
y = data[target]

# Train/Test/Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisation and fitting
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Evaluating the performance of the model using MSE and R2
prediction = model.predict(X_test)

mse = mean_squared_error(y_test, prediction)
r2 = r2_score(y_test, prediction)

print("Mean Squared Error (MSE): ", mse)
print("R-squared (R2): ", r2)

#Feature Weight Plotting
weights = model.feature_importances_
indices = weights.argsort()[::-1]
features = X.columns

sorted_weights = [weights[i] for i in indices]
sorted_features = [features[i] for i in indices]

plt.barh(range(len(indices)), sorted_weights)
plt.yticks(range(len(indices)), sorted_features)
plt.ylabel('Weight Score')
plt.title('Feature Importance - Decision Tree')
plt.show()
