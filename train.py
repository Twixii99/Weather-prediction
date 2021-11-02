
# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import pickle

from sklearn.metrics import mutual_info_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Important Variables
model_name = 'xgb_model'
model_file = f'model_{model_name}.bin'
data_file = 'weatherAUS.csv'

# Data Loading
df = pd.read_csv(data_file)

del df['Date']

print(f'Succesfully Loading the data {data_file}')

# Data Cleaning 
df.columns = df.columns.str.lower()

categorical_variables = ['location',
  'windgustdir',
  'winddir9am',
  'winddir3pm',
  'raintoday']

numerical_variables = ['mintemp',
  'maxtemp',
  'rainfall',
  'evaporation',
  'sunshine',
  'windgustspeed',
  'windspeed9am',
  'windspeed3pm',
  'humidity9am',
  'humidity3pm',
  'pressure9am',
  'pressure3pm',
  'cloud9am',
  'cloud3pm',
  'temp9am',
  'temp3pm']

for col in categorical_variables:
  df[col] = df[col].str.lower()

#  Explaratory Data Analysis

## 5.0 The target variable analysis
df = df.drop(df[df.raintomorrow.isnull()].index)
df = df.reset_index(drop=True)

df['raintomorrow'] = (df['raintomorrow'] == 'Yes').astype(int)

## 5.1 Categorical feature analysis

### 5.1.1 Dealing with the missing values

for col in categorical_variables:
  df[col] = df.groupby(df['raintomorrow'])[col].apply(lambda f: f.fillna(f.mode().values[0]))

df['raintoday'] = (df['raintoday'] == 'yes').astype(int).astype(str)

## 5.2 Numerical feature analysis

### 5.2.1 Univariate analysis

uniformally_distributed_variables = ['mintemp', 'maxtemp', 
                                     'sunshine', 'windgustspeed', 
                                     'windspeed3pm', 'humidity9am', 
                                     'humidity3pm', 'pressure9am',
                                     'pressure3pm', 'temp9am', 
                                     'temp3pm'
                                     ]
for col in uniformally_distributed_variables:
  df[col] = df.groupby(df['raintomorrow'])[col].apply(lambda f: f.fillna(f.mean()))

#### More Checking the other numerical variables distributions
right_skewed_distributed_variables = ['rainfall', 'evaporation', 'windspeed9am', 'cloud9am', 'cloud3pm']

for col in right_skewed_distributed_variables:
  df[col] = df.groupby(df['raintomorrow'])[col].apply(lambda f: f.fillna(f.median()))

# 6. Model Trainings

## 6.1 Train, Validation, Test datasets extraction
full_train, test = train_test_split(df, test_size=0.2, shuffle=True, random_state=11)
train, validation = train_test_split(full_train, test_size=0.25, shuffle=True, random_state=11)

train = train.reset_index(drop=True)
validation = validation.reset_index(drop=True)
test = test.reset_index(drop=True)
full_train = full_train.reset_index(drop=True)

y_train = train['raintomorrow'].values
y_val = validation['raintomorrow'].values
y_test = test['raintomorrow'].values
y_full_train = full_train['raintomorrow'].values

del train['raintomorrow']
del validation['raintomorrow']
del test['raintomorrow']
del full_train['raintomorrow']

## 6.2 One-hot encoding
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train.to_dict(orient='records'))
X_val = dv.transform(validation.to_dict(orient='records'))

dv2 = DictVectorizer(sparse=False)
X_full_train = dv2.fit_transform(full_train.to_dict(orient='records'))
X_test = dv2.transform(test.to_dict(orient='records'))

# 6.6 XGBoosting
D_train = xgb.DMatrix(X_train, label=y_train, feature_names=dv.get_feature_names())
D_val = xgb.DMatrix(X_val, label=y_val, feature_names=dv.get_feature_names())
D_full_train = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=dv2.get_feature_names())
D_test = xgb.DMatrix(X_test, label=y_test, feature_names=dv2.get_feature_names())

xgb_params = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

xgb_model = xgb.train(xgb_params, D_train, num_boost_round=100)
y_pred = xgb_model.predict(D_val)
print('The ROC_AUC score of validation data = ', roc_auc_score(y_val, y_pred))

xgb_model_test = xgb.train(xgb_params, D_full_train, num_boost_round=100)
y_pred_test = xgb_model_test.predict(D_test)
print('The ROC_AUC score of test data = ', roc_auc_score(y_test, y_pred_test))

# 7. Model Saving and Loading
with open(model_file, 'wb') as m_out:
  pickle.dump((dv2, xgb_model_test), m_out)

  print(f'The model is saved to {model_file}')
