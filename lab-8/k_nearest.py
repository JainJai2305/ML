from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
Y = iris.target

print('sepal-length','sepal-width','petal-length','petal-width')
print(X)
print('target')
print(Y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Training the model with Nearest nighbors K=3
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)
matrix =confusion_matrix(y_test,y_pred) 
print(" Confusion matrix:\n",matrix)
print(" Correct predicition",accuracy_score(y_test,y_pred))
print(" Wrong predicition",(1-accuracy_score(y_test,y_pred)))
print(' Accuracy Metrics')
print(classification_report(y_test,y_pred))
