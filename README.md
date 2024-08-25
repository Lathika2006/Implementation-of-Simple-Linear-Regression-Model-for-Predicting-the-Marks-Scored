## Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: LATHIKA L J
RegisterNumber: 212223220050


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
 
Y_pred

Y_test

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```

## Output:

![Screenshot 2024-08-24 135442](https://github.com/user-attachments/assets/9ee37ac3-13ec-4ad6-8ca7-c6a64a4bc359)

![Screenshot 2024-08-24 135403](https://github.com/user-attachments/assets/9095779b-3fa1-4674-a649-ac7a4041f72e)

![Screenshot 2024-08-24 135352](https://github.com/user-attachments/assets/c69b81ed-e356-4cff-8e97-292eab36bda2)

![Screenshot 2024-08-24 135333](https://github.com/user-attachments/assets/d83fc2bd-b302-4fcb-bcd2-c6c300085e86)

![Screenshot 2024-08-24 135325](https://github.com/user-attachments/assets/53d861c2-7826-4e5b-868c-1c6a866c577f)

![Screenshot 2024-08-24 135314](https://github.com/user-attachments/assets/fcda8e7b-2497-4f5b-a33f-ce2934999f8e)

![Screenshot 2024-08-24 135303](https://github.com/user-attachments/assets/6fb0f4c9-32f1-473b-aa66-8d05f6eb0cdb)

![Screenshot 2024-08-24 135249](https://github.com/user-attachments/assets/7643448b-1d3a-4ed6-93f9-6c4c58988280)

![Screenshot 2024-08-24 135203](https://github.com/user-attachments/assets/8a18c164-7ca7-4c7b-9070-52d0984922bf)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
