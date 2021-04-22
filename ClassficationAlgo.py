import sklearn
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("car.data")
print(data)
# change the string values in to integer values
labelE = preprocessing.LabelEncoder()
#  get an entire column and change the data type in to integer--> each are numpy arrays
buying = labelE.fit_transform(list(data["buying"]))
maintenance = labelE.fit_transform(list(data["maintenance"]))
door = labelE.fit_transform(list(data["door"]))
persons = labelE.fit_transform(list(data["persons"]))
lug_boot = labelE.fit_transform(list(data["lug_boot"]))
cls = labelE.fit_transform(list(data["class"]))
safety = labelE.fit_transform(list(data["safety"]))

predict = "class"

#  x is our feature
#  y is our label
x = list(zip(buying, maintenance, door, persons, lug_boot, safety))
y = list(cls)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, train_size=0.1)
#  the more number of neighbors,  the more the accuracy
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)
predicted = model.predict(x_test)
#  print(predicted)
names = ["unacc", "acc", "good", "vgood"]
for i in range(len(predicted)):
    print("predicted: ", names[predicted[i]], "Data: ", x_test[i], "Actual: ", names[y_test[i]])
