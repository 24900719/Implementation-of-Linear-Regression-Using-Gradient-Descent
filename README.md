<img width="741" height="163" alt="image" src="https://github.com/user-attachments/assets/4fb2632a-fb00-4683-8a49-b9acb5bb63ee" /># Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
<img width="688" height="710" alt="image" src="https://github.com/user-attachments/assets/ba8b06f8-d5a9-40f6-ac3d-8917bf3f7507" />


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: K SARANYA
RegisterNumber:  212224040298
*/
```!
/*

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
print(data.head())
print("\n")
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print("\n")
print(X1_Scaled)
print("\n")
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
    
*/
## Output:
![linear regression using gradient descent](sam.png)
<img width="741" height="163" alt="image" src="https://github.com/user-attachments/assets/c3935ece-5445-43f0-8172-22b443fa7fc7" />
<img width="301" height="723" alt="image" src="https://github.com/user-attachments/assets/e98f4c48-37c0-4122-90eb-dd1f8e18a85e" />
<img width="382" height="783" alt="image" src="https://github.com/user-attachments/assets/4711237b-7c95-47ed-93d1-97023222ea7f" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
