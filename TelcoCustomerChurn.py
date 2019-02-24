#Artificial Neural Network

#Data PreProcessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Importing the dataset
dataset = pd.read_csv("TelcoCustomerChurn.csv")
#dataset["Internet"] = np.where(dataset["InternetService"] != "No",1,0)
#dataset["AllServices"] = (dataset["OnlineSecurity"] + dataset["OnlineBackup"] + dataset["DeviceProtection"] + 
#       dataset["TechSupport"] + dataset["PhoneService"] + dataset["MultipleLines"] + dataset["Internet"] + 
#       dataset["StreamingTV"] + dataset["StreamingMovies"]).astype(int) 
#dataset["Family"] = np.where(dataset["Partner"]+dataset["Dependents"] != 0, 1,0)

#Drop unnecessary columns - gender, TotalCharges
#dataset = dataset.drop(["gender","Partner","Dependents","PhoneService","MultipleLines",
#                        "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
#                        "TechSupport","StreamingTV","StreamingMovies","MonthlyCharges","TotalCharges",
#                        "Internet"],axis=1)
dataset = dataset.drop(["gender","TotalCharges"],axis=1)
#Re-arranging the columns
dataset = dataset[['Contract','PaymentMethod',"InternetService","PhoneService","MultipleLines","OnlineSecurity",
                   "OnlineBackup", "DeviceProtection", "TechSupport","StreamingTV","StreamingMovies",
                   'SeniorCitizen','tenure','PaperlessBilling', "Partner","Dependents",
                   "MonthlyCharges",'Churn']]

#Correlation Matrix
corr_matrix = dataset.corr()

#Exploring dataset
#Gender - Does not impact churn
#sns.barplot(x="gender",y="Churn", data = dataset)
#Senior Citizen - Senior citizens tend to churn more
#sns.barplot(x="SeniorCitizen",y="Churn",data=dataset)
#Partners - Customers without partners tend to churn more
#sns.barplot(x="Partner",y="Churn",data=dataset)
#Dependents - Customers without dependents tend to churn more
#sns.barplot(x="Family",y="Churn",data=dataset)
#Tenure - Customers with lower tenure tend to churn more
#sns.boxplot(x="Churn",y="tenure",data=dataset)
#Services that generate loyalty - OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport
#sns.barplot(x="DeviceProtection",y="Churn",data=dataset)
#LoyaltyServices
#sns.barplot(x="Internet",y="Churn",data=dataset)
#Service that may harm loyalty - PhoneService, MultipleLines, InternetService, StreamingTV, StreamingMovies
#sns.barplot(x="StreamingTV",y="Churn",data=dataset)
#ReducesLoyalties
#sns.barplot(x="AllServices",y="Churn",data=dataset)
#Contract - Month-to-Month contract customers are more likely to churn
#sns.barplot(x="Contract",y="Churn",data=dataset)
#PaperlessBilling - Customers enrolled for paperless billing have a higher likelihood to churn
#sns.barplot(x="PaperlessBilling",y="Churn",data=dataset)
#Payment Method - Customers enrolled for Electronic Check are more likely to churn
#sns.barplot(x="PaymentMethod",y="Churn",data=dataset)
#Monthly Charges - Higher the Monthly Charges, more likely customer is to churn
#sns.boxplot(x="Churn",y="MonthlyCharges",data=dataset)

#Creating X and Y variables
X = dataset.iloc[:,0:17].values
y = dataset.iloc[:,17].values

#Treat categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Encode categorical data as labels
labelencoder_contract = LabelEncoder()
X[:,0] = labelencoder_contract.fit_transform(X[:,0])
labelencoder_payment = LabelEncoder()
X[:,1] = labelencoder_payment.fit_transform(X[:,1])
labelencoder_iservice = LabelEncoder()
X[:,2] = labelencoder_iservice.fit_transform(X[:,2])
#Convert into dummy variables 
onehotencoder = OneHotEncoder(categorical_features=[0,1,2])
X = onehotencoder.fit_transform(X).toarray()
#Remove one column out of each categorical variable - Dummy variable trap
X = X[:,[1,2,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19]]

#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
#Use feature scaling of the training set
X_test = sc.transform(X_test)

#importing Keras libraries and packages
import keras
#ANN model building layer by layer
from keras.models import Sequential
from keras.layers import Dense
#Regularization to reduce overfitting - Disables neurons at random
from keras.layers import Dropout
#Initializing the ANN - Building the framework
classifier = Sequential()
classifier.add(Dense(units=9,kernel_initializer="uniform",activation="relu",input_dim=17))
#classifier.add(Dropout(rate=0.1))
classifier.add(Dense(units=9,kernel_initializer="uniform",activation="relu"))
classifier.add(Dropout(rate=0.1))
#classifier.add(Dense(units=5,kernel_initializer="uniform",activation="relu"))
#classifier.add(Dropout(rate=0.1))
classifier.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid"))

#Compiling the ANN
#classifier.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=["accuracy"])
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

#Fitting the ANN to the training set
classifier.fit(x=X_train,y=y_train,batch_size=32,epochs=25)

#Predicting the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
(cm[0,0] + cm[1,1])/cm.sum()
#Evaluating, Improving and Tuning the ANN
#Evaluating the ANN
#Keras wrapper for sklearn for applying cross validation on ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
#Function that builds the model
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=9,kernel_initializer="uniform",activation="relu",input_dim=17))
    #classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units=9,kernel_initializer="uniform",activation="relu"))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units=5,kernel_initializer="uniform",activation="relu"))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid"))
    classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    return classifier
#Keras Classifier
classifier = KerasClassifier(build_fn = build_classifier,batch_size=32,epochs=25)
#Build the model with 10 cross validations
accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train, cv = 10,verbose=1)
accuracies.mean()
#Tuning parameters
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
#Function that builds the model
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=9,kernel_initializer="uniform",activation="relu",input_dim=17))
    classifier.add(Dense(units=9,kernel_initializer="uniform",activation="relu"))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units=5,kernel_initializer="uniform",activation="relu"))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid"))
    classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    return classifier
#Keras Classifier
classifier = KerasClassifier(build_fn = build_classifier)
#Dictionary for Grid Search
parameters = {"batch_size":[10,20],
              "epochs":[10,50],
              "optimizer":["adam","rmsprop"]}
grid_search = GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10)
grid_search = grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
