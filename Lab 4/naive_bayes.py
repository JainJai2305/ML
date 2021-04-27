import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

df = pd.read_csv("/content/pima_indian.csv")
feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_names = ['diabetes']
X = df[feature_col_names].values 
y = df[predicted_class_names].values
print(df.head)
xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.33)
print ('\nThe total number of Training Data:',ytrain.shape)
print ('The total number of Test Data:',ytest.shape)

# <bound method NDFrame.head of      num_preg  glucose_conc  diastolic_bp  ...  diab_pred  age  diabetes
# 0           6           148            72  ...      0.627   50         1
# 1           1            85            66  ...      0.351   31         0
# 2           8           183            64  ...      0.672   32         1
# 3           1            89            66  ...      0.167   21         0
# 4           0           137            40  ...      2.288   33         1
# ..        ...           ...           ...  ...        ...  ...       ...
# 763        10           101            76  ...      0.171   63         0
# 764         2           122            70  ...      0.340   27         0
# 765         5           121            72  ...      0.245   30         0
# 766         1           126            60  ...      0.349   47         1
# 767         1            93            70  ...      0.315   23         0

# [768 rows x 9 columns]>

# The total number of Training Data: (514, 1)
# The total number of Test Data: (254, 1)

clf = GaussianNB().fit(xtrain,ytrain.ravel())
predicted = clf.predict(xtest)
predictTestData= clf.predict([[6,148,72,35,0,33.6,0.627,50]])
print('\nConfusion matrix')
print(metrics.confusion_matrix(ytest,predicted))
print('\nAccuracy of the classifier:',metrics.accuracy_score(ytest,predicted))
print('The value of Precision:', metrics.precision_score(ytest,predicted))
print('The value of Recall:', metrics.recall_score(ytest,predicted))
print("Predicted Value for individual Test Data:", predictTestData)


# Confusion matrix
# [[156  16]
#  [ 35  47]]

# Accuracy of the classifier: 0.7992125984251969
# The value of Precision: 0.746031746031746
# The value of Recall: 0.573170731707317
# Predicted Value for individual Test Data: [1]
