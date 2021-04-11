import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

train = pd.read_csv('./project_data/train.csv')
test = pd.read_csv('./project_data/test.csv')
gender_submission = pd.read_csv('./project_data/gender_submission.csv')

train = train.drop(columns=['Ticket', 'Cabin', 'Embarked', 'PassengerId', 'Name'])
train[['Sex']] = OrdinalEncoder().fit_transform(train[['Sex']])
train['Age'].fillna(value=train['Age'].median(), inplace=True)
train['Fare'].fillna(value=train['Fare'].median(), inplace=True)
train['Age'] = StandardScaler().fit_transform(train['Age'].values.reshape(-1, 1))
train['Fare'] = StandardScaler().fit_transform(train['Fare'].values.reshape(-1, 1))

X = train.drop(columns=['Survived'])
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)

model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, random_state=31),
                             n_estimators=200, algorithm="SAMME.R", learning_rate=0.5)
model.fit(X_train,y_train)
model_predictions = model.predict(X_test)

test = test.drop(columns=['Ticket', 'Cabin', 'Embarked', 'PassengerId', 'Name'])
test[['Sex']] = OrdinalEncoder().fit_transform(test[['Sex']])
test['Age'].fillna(value=test['Age'].median(), inplace=True)
test['Fare'].fillna(value=test['Fare'].median(), inplace=True)
test['Age'] = StandardScaler().fit_transform(test['Age'].values.reshape(-1, 1))
test['Fare'] = StandardScaler().fit_transform(test['Fare'].values.reshape(-1, 1))

predictions = model.predict(test)

submission = pd.DataFrame({'PassengerId':gender_submission['PassengerId'],'Survived':predictions})
filename = 'First_try.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)