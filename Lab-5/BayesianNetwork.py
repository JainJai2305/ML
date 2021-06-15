import numpy as np
import pandas as pd
import csv
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

trainingData = pd.read_csv('/content/bayesian-dataset.csv')
trainingData = trainingData.replace('?',np.nan)
print('The sample instances from the dataset are:')
print(trainingData.head())
print('\n Attributes and datatypes: ')
print(trainingData.dtypes)

#The sample instances from the dataset are:
#    age  sex  cp  trestbps  chol  ...  oldpeak  slope  ca  thal  heartdisease
# 0   63    1   1       145   233  ...      2.3      3   0     6             0
# 1   67    1   4       160   286  ...      1.5      2   3     3             2
# 2   67    1   4       120   229  ...      2.6      2   2     7             1
# 3   37    1   3       130   250  ...      3.5      3   0     3             0
# 4   41    0   2       130   204  ...      1.4      1   0     3             0

# [5 rows x 14 columns]

#  Attributes and datatypes: 
# age               int64
# sex               int64
# cp                int64
# trestbps          int64
# chol              int64
# fbs               int64
# restecg           int64
# thalach           int64
# exang             int64
# oldpeak         float64
# slope             int64
# ca               object
# thal             object
# heartdisease      int64
# dtype: object

model = BayesianModel([('age','heartdisease'),('sex','heartdisease'),('exang','heartdisease'),('cp','heartdisease'),('heartdisease','restecg'),('heartdisease','chol')])
print('\n Learning CPD using Maximum likelihood estimators')
model.fit(trainingData,estimator=MaximumLikelihoodEstimator)
print('\n Inferencing with Bayesian Network:')
HeartDiseasetest_infer = VariableElimination(model)
print('\n 1.Probability of HeartDisease given evidence = restecg (Rest ECG): 1')
q1 = HeartDiseasetest_infer.query(variables = ['heartdisease'], evidence={'restecg':1})
print(q1)

# 1.Probability of HeartDisease given evidence = restecg (Rest ECG): 1
# +-----------------+---------------------+
# | heartdisease    |   phi(heartdisease) |
# +=================+=====================+
# | heartdisease(0) |              0.1012 |
# +-----------------+---------------------+
# | heartdisease(1) |              0.0000 |
# +-----------------+---------------------+
# | heartdisease(2) |              0.2392 |
# +-----------------+---------------------+
# | heartdisease(3) |              0.2015 |
# +-----------------+---------------------+
# | heartdisease(4) |              0.4581 |
# +-----------------+---------------------+

print('\n 2.Probability of HeartDisease given evidence = chol (Cholestorol): 100 ')
q2 = HeartDiseasetest_infer.query(variables = ['heartdisease'], evidence={'chol':100})
print(q2)
# 2.Probability of HeartDisease given evidence = chol (Cholestorol): 100 
# +-----------------+---------------------+
# | heartdisease    |   phi(heartdisease) |
# +=================+=====================+
# | heartdisease(0) |              1.0000 |
# +-----------------+---------------------+
# | heartdisease(1) |              0.0000 |
# +-----------------+---------------------+
# | heartdisease(2) |              0.0000 |
# +-----------------+---------------------+
# | heartdisease(3) |              0.0000 |
# +-----------------+---------------------+
# | heartdisease(4) |              0.0000 |
# +-----------------+---------------------+
