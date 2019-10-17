#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:37:28 2019

@author: aaman10
"""


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb

TRAIN_FILE = '/Users/aaman10/Desktop/Kaggle/NFL/train.csv'

TRAIN_ = pd.read_csv(TRAIN_FILE)
TRAIN_NUM = TRAIN_._get_numeric_data()
TRAIN_NUM_FILLED = TRAIN_NUM.fillna(TRAIN_NUM.mean())

del TRAIN_, TRAIN_NUM
X = TRAIN_NUM_FILLED.drop('Yards', axis=1)
Y = TRAIN_NUM_FILLED['Yards']

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.2,
                                                    stratify=Y)

DT = DecisionTreeClassifier(max_depth=2)
DT.fit(X_TRAIN, Y_TRAIN)
Y_PRED = DT.predict(X_TEST)
print(accuracy_score(Y_TEST, Y_PRED))


LGBM = lgb.LGBMClassifier()
LGBM.fit(X, Y)
YY = LGBM.predict(X_TEST)
print(accuracy_score(Y_TEST, YY))

FEATURE_IMP = pd.DataFrame(zip(LGBM.feature_importances_,
                               X.columns), columns=['Value','Feature']).sort_values(by='Value', ascending=False)
print(FEATURE_IMP)