import numpy as np
import pandas as pd
from linear_regression import LinearRegression

data = pd.read_csv('insurance.csv')

#Preprocess the data by converting vars to dummy vars(one-hot)
data = pd.get_dummies(data, drop_first=True)

# Add a bias feature (column of ones)
data.insert(0, 'bias', 1)

#Add bias term to the data
#x_bias = np.c_[np.ones(data.shape[0]), data]

#Features and target, drop charges since it is target
X = data.drop('charges', axis=1).values
y = data['charges'].values

#Shuffle rows of data
#Split into two subsets: 2/3 for training, 1/3 for validation
#Train the model 
np.random.seed(0)
indices = np.random.permutation(len(X))
train_size = int(len(X) * 2 / 3)
train_indices, val_indices = indices[:train_size], indices[train_size:]

#Assign shuffeled data into training and validation sets
X_train, X_val = X[train_indices], X[val_indices]
y_train, y_val = y[train_indices], y[val_indices]

#Load linear regression model and fit it to training data
model = LinearRegression()
model.fit(X_train, y_train)

#Have model predict on training and validation sets
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

print("Training RMSE:", model.rmse(y_train, y_train_pred))
print("Validation RMSE:", model.rmse(y_val, y_val_pred))
print("Training SMAPE:", model.smape(y_train, y_train_pred))
print("Validation SMAPE:", model.smape(y_val, y_val_pred))
