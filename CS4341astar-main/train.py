import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from math import sqrt

data = pd.read_csv("Dillon2_csv.csv")

yTrain = data.cost_from_node.values
# xTrain = data.get("euclidean_distance")
xTrain = data.get(["horizontal_distance", "vertical_distance", "euclidean_distance"])

plt.scatter(xTrain["euclidean_distance"], yTrain, facecolor="None", edgecolors='k', alpha=.3)

# training = data.sample(frac=.5, replace=False, random_state=1)
# test = data.drop(training.index)

xPred = xTrain["euclidean_distance"]
model = LinearRegression()
model.fit(xTrain, yTrain)
xTest = np.linspace(xPred.min(), xPred.max(), 100)
pred = model.predict(xTest)
mse = mean_squared_error(yTrain, pred)

print("MSE = " + str(mse))
plt.figure()
plt.plot(xTrain["euclidean_distance"], pred, label="max_depth=2", linewidth=2)
plt.title("Random Forest Regression")
plt.show()