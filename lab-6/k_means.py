import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd
import numpy as np

iris = datasets.load_iris()
X = pd.DataFrame(iris.data)
X.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']

y = pd.DataFrame(iris.target)
y.columns = ['Targets']

print(X.head())
print(y.head())
#    Sepal_Length  Sepal_Width  Petal_Length  Petal_Width
# 0           5.1          3.5           1.4          0.2
# 1           4.9          3.0           1.4          0.2
# 2           4.7          3.2           1.3          0.2
# 3           4.6          3.1           1.5          0.2
# 4           5.0          3.6           1.4          0.2
#    Targets
# 0        0
# 1        0
# 2        0
# 3        0
# 4        0

model = KMeans(n_clusters=3)
model.fit(X)
plt.figure(figsize=(14,7))

colormap = np.array(['red', 'lime', 'black'])
# Plot the Original Classifications
plt.subplot(1, 2, 1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets], s=40)
plt.title('Real Classification')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
# Plot the Models Classifications
plt.subplot(1, 2, 2)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model.labels_], s=40)
plt.title('K Mean Classification')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
print('The accuracy score of K-Mean: ',sm.accuracy_score(y, model.labels_))
print('The Confusion matrixof K-Mean:\n ',sm.confusion_matrix(y, model.labels_))
# The accuracy score of K-Mean:  0.8933333333333333
# The Confusion matrixof K-Mean:
#   [[50  0  0]
#  [ 0 48  2]
#  [ 0 14 36]]
