import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.metrics import explained_variance_score, mean_squared_error
from math import sqrt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split

# load the boston dataset
boston = datasets.load_boston(return_X_y=False)

# defining feature matrix(X) and response vector(y)
X = boston.data
y = boston.target

# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# create linear regression object
reg = LinearRegression()

# train the model using the training sets
reg.fit(X_train, y_train)

# regression coefficients
print('Coefficients: \n', reg.coef_)

# variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(X_test, y_test)))


y_pred = reg.predict(X_test)
n_errors = (y_pred != y_test).sum()




print('explained_variance_score: {}'.format(explained_variance_score(y_test, y_pred)))

print("explained_variance_score with uniform_average: {}".format(explained_variance_score(y_test, y_pred, multioutput="uniform_average")))



rms = sqrt(mean_squared_error(y_test, y_pred))

print("mean squared error: {}".format(rms))

intercept_gh = reg.intercept_

print("intercept value: {}".format(intercept_gh))




style.use("bmh")


plt.xlabel("y prediction")
plt.ylabel("y test")
plt.title("y predicted value and  y_test value")
plt.scatter(y_test, y_pred, color = "red", s = 19)
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), color = "pink", linewidth=6)
plt.show()




print("y prediction by predict function on X_test: {}".format(y_pred))

print('{}: {}'.format("linear regression number of  error", n_errors))


# plot for residual error

## setting plot style
plt.style.use('fivethirtyeight')

## plotting residual errors in training data
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
            color="red", s=19, label='Train data')

## plotting residual errors in test data
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
            color="grey", s=19, label='Test data')

## plotting line for zero residual error
plt.hlines(y=0, xmin=0, xmax=50, linewidth=6, color = "yellow")

## plotting legend
plt.legend(loc='upper right', prop={'size': 10})

## plot title
plt.title("Residual errors", color = "brown")

## function to show plot
plt.show()
















