#Imports
import xgboost as xgb

#Configuration
Seed = 12
Task ='classification'
LabelColumn = 'target'
IterationNumber = 50
Classifier = xgb.XGBClassifier()
NumClasses = 3
LabelColumn = 'target'
P_Value = 0.05
Percentile = None