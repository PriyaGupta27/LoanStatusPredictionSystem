# the libraries are imported
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
import joblib

#reading the dataset
train = pd.read_csv("train.csv")

# data preprocessing
le = LabelEncoder()
train['LoanAmount'].fillna(train['LoanAmount'].mean(), inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].median(), inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train.drop(['Loan_ID'], axis=1, inplace=True)
train['LoanAmount'] = np.log(train['LoanAmount'])
col_to_encode = ['Gender','Dependents','Married','Education','Self_Employed','Property_Area']
train[col_to_encode] = train[col_to_encode].apply(le.fit_transform)
train['Loan_Status'] = train['Loan_Status'].map({'N': 0, 'Y': 1}).astype(int)
X = train.drop('Loan_Status', axis=1)
Y = train.Loan_Status

#handling imbalanced data
sm = SMOTE(random_state=0)
X, Y = sm.fit_resample(X, Y)

#split the dataset in training and testing
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#creating the model
classifier = GradientBoostingClassifier(
    n_estimators = 60,
    learning_rate=0.25,
    random_state = 0
)  
classifier.fit(x_train,y_train)
filename = 'classifier_model.joblib'
joblib.dump(classifier,filename)