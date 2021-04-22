import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle  #  to save our best accuracy model
from matplotlib import style

#  Linear regression algorithm
data = pd.read_csv("student-mat.csv", sep=";")
#  trimming off some of the attributes (choosing only the ones with floating values)
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"
X = np.array(data.drop([predict], 1))
#  print('X\n',X)
Y = np.array(data[predict])
#  print('Y:\n',Y)
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
best = 0
#  keep training till I get the best accuracy and save
for _ in range(30):

    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
    lModel = linear_model.LinearRegression()
    lModel.fit(X_train, Y_train)
    accuracyL = lModel.score(X_test, Y_test)
    print('accuracy level: \n', accuracyL)

    if accuracyL > best:
        best = accuracyL
        with open("studentmodel.pkl", "wb") as f:
           pickle.dump(lModel, f)


pickle_in = open("studentmodel.pkl", "rb")
lModel = pickle.load(pickle_in)
print('Coefficient: \n', lModel.coef_)
print('Intercept: \n', lModel.intercept_)
print('___________________________________________________________')
predictions = lModel.predict(X_test)
#  print(predictions)
#  print(predictions.shape)
for i in range(len(predictions)):
    print(predictions[i], X_test[i], Y_test[i])

#  the x-axis data, also shows the corelations b/n this and  the the final grade
p1 = "G1"
p2 = "G2"
p3 = "studytime"
p4 = "failures"
p5 = "absences"
# plotting the data in grid
style.use("ggplot")
#  set up scatter plot
pyplot.scatter(data[p5], data["G3"])  # G3 is the label
pyplot.xlabel(p5)
pyplot.ylabel("Final Grade Prediction")
pyplot.show()
