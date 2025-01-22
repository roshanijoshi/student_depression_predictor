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

np.random.seed(42)
data = pd.read_csv("/content/Student Depression Dataset.csv")
data = data.dropna()
data

X=data[['Work/Study Hours','Financial Stress']]
Y=data['Depression']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

logistic_model = LogisticRegression()
logistic_model.fit(X_train, Y_train)

joblib.dump(logistic_model, "student_depression_dataset.pkl")
Y_pred = logistic_model.predict(X_test)
accuracy_score(Y_test,Y_pred)
