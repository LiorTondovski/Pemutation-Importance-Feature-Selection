#Configuration
import xgboost as xgb

Distributions = ['norm', 'lognorm', 'gamma']
Seed = 12
Task ='classification'
IterationNumber = 50
Classifier = xgb.XGBClassifier()
NumClasses = 3
LabelColumn = 'target'
P_Value_Threshold = 0.05
MinimalMatchingFeaturesRatio=0.8
UseEstimation=False