# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
import pandas module and import the required data set.
Find the null values and count them.
Count number of left values.
From sklearn import LabelEncoder to convert string values to numerical values.
From sklearn.model_selection import train_test_split.
Assign the train dataset and test dataset.
From sklearn.tree import DecisionTreeClassifier.
Use criteria as entropy.
From sklearn import metrics.
Find the accuracy of our model and predict the require values.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Rithish R
RegisterNumber:  212224040278
import pandas as pd
data = pd.read_csv("Employee.csv")
data
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print(confusion)
*/
```

## Output:
## data:
<img width="1036" height="618" alt="image" src="https://github.com/user-attachments/assets/1d626542-6f21-4a04-a523-2e702b5cf04d" />

## accuracy:
<img width="1231" height="76" alt="image" src="https://github.com/user-attachments/assets/8f893e44-0fa1-405c-8aa2-b8f666fc6962" />

## predict:
<img width="1038" height="94" alt="image" src="https://github.com/user-attachments/assets/33e07d27-ab82-40d1-9db8-53b7c0cc8029" />

## classification_report:
<img width="342" height="95" alt="image" src="https://github.com/user-attachments/assets/aa177acc-7b17-4d85-a7f2-0347e76ccd72" />

## confusion_matix:
<img width="1004" height="279" alt="image" src="https://github.com/user-attachments/assets/ac16f20c-6925-4fcc-b9cf-92153044f2ed" />





## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
