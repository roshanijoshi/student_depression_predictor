import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

np.random.seed(42)
data = pd.read_csv("/content/Student Depression Dataset.csv")
data = data.dropna()
data

X=data[['Work/Study Hours','Financial Stress','Academic Pressure','Work Pressure','Age']]
Y=data['Depression']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train


logistic_model = LogisticRegression()
logistic_model.fit(X_train, Y_train)

joblib.dump(logistic_model, "student_depression_dataset.pkl")
Y_pred = logistic_model.predict(X_test)
accuracy_score(Y_test,Y_pred)
accuracy_score(Y_test, Y_pred)*100

logistic_model.score(X_train, Y_train)*100

logistic_model.score(X_test, Y_test)*100


precision = precision_score(Y_test, Y_pred, average='weighted')
precision*100

recall= recall_score(Y_test,Y_pred)
recall*100

f1=f1_score(Y_test,Y_pred)
f1*100

classification=classification_report(Y_test,Y_pred)
classification
