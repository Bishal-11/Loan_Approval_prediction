LOANS are the major requirement of the modern world. By this only, Banks get a major part of the total profit. It is beneficial for students to manage their education and living expenses, and for people to buy any kind of luxury like houses, cars, etc.
But when it comes to deciding whether the applicant’s profile is relevant to be granted with loan or not. Banks have to look after many aspects.So, here we will be using Machine Learning with Python to ease their work and predict whether the candidate’s profile is relevant or not using key features like Marital Status, Education, Applicant Income, Credit History, etc.

Loan Approval Prediction using Machine Learning

# 1. Importing Libraries and Dataset
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

data = pd.read_csv("/content/loan_approval_dataset.csv") 

# 2.Data Preprocessing and Visualization
obj = (data.dtypes == 'object') 
print("Categorical variables:",len(list(obj[obj].index)))4

import os
print(os.getcwd())

data = pd.read_csv('loan_approval_dataset.csv')

# Now you can use the drop method

# Dropping Loan_ID column 
data.drop(['loan_id'],axis=1,inplace=True)

# 3. Visualize all the unique values in columns using barplot.
#import matplotlib.pyplot as plt
#import seaborn as sns

obj = (data.dtypes == 'object')
object_cols = list(obj[obj].index)
plt.figure(figsize=(18, 36))
index = 1

for col in object_cols:
    y = data[col].value_counts()
    plt.subplot(11, 4, index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index), y=y)
    index += 1
    
   # As all the categorical values are binary so we can use Label Encoder for all such columns and the values will change into int datatype.
# Import label encoder
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
obj = (data.dtypes == 'object')
for col in list(obj[obj].index):
    data[col] = label_encoder.fit_transform(data[col])
    # To find the number of columns with 
# datatype==object 
obj = (data.dtypes == 'object') 
print("Categorical variables:",len(list(obj[obj].index)))
plt.figure(figsize=(12,6)) 

sns.heatmap(data.corr(),cmap='BrBG',fmt='.2f', 
			linewidths=2,annot=True)
# 4. Now we will use Catplot to visualize the plot for the Gender, and Marital Status of the applicant.
sns.catplot(x="Gender", y="Married", data=data)
plt.show()
# find out if there is any missing values in the dataset using below code.

for col in data.columns:
    data[col] = data[col].fillna(data[col].mean())
# 5. Splitting Dataset 
from sklearn.model_selection import train_test_split 

X = data.drop([' loan_status'],axis=1) 
Y = data[' loan_status'] 
X.shape,Y.shape 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
													test_size=0.4, 
													random_state=1) 
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

# 6. Model Training and Evaluation

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression 

from sklearn import metrics 

knn = KNeighborsClassifier(n_neighbors=3) 
rfc = RandomForestClassifier(n_estimators = 7, 
							criterion = 'entropy', 
							random_state =7) 
svc = SVC() 
lc = LogisticRegression() 

# making predictions on the training set 

for clf in (rfc, knn, svc,lc): 
	clf.fit(X_train, Y_train) 
	Y_pred = clf.predict(X_train) 
	print("Accuracy score of ", 
		clf.__class__.__name__, 
		"=",100*metrics.accuracy_score(Y_train, 
										Y_pred))

# making predictions on the testing set 

for clf in (rfc, knn, svc,lc): 
	clf.fit(X_train, Y_train) 
	Y_pred = clf.predict(X_test) 
	print("Accuracy score of ", 
		clf.__class__.__name__,"=", 
		100*metrics.accuracy_score(Y_test, 
									Y_pred))
data.isna().sum()

# 7. Conclusion
Random Forest Classifier is giving the best accuracy with an accuracy score of 96% for the testing dataset.

And to get much better results ensemble learning techniques like Bagging and Boosting can also be used.
