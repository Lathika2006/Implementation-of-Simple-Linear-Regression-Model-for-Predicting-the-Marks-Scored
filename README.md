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
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: LATHIKA L J
RegisterNumber: 212223220050
*/
```
```
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

![image](https://github.com/user-attachments/assets/0386eb01-8518-493d-babf-f98182a43264)
![image](https://github.com/user-attachments/assets/6c437fb2-b872-495d-9243-905b92f61aa0)
![image](https://github.com/user-attachments/assets/31ff54f8-a4c6-4824-a844-4b7101b2efde)
![image](https://github.com/user-attachments/assets/e542cc0f-d053-49da-ad1f-28e735e652de)
![image](https://github.com/user-attachments/assets/b9c54bf0-f151-471a-bdf3-6226e3b9fb9a)
![image](https://github.com/user-attachments/assets/71e25605-eb27-4e81-9661-7b65f14b94ac)
![Screenshot 2024-08-24 135442](https://github.com/user-attachments/assets/250711cf-8c6b-4494-b800-64b253f0f94b)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
