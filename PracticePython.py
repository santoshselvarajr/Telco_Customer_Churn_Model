#Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Train Test Split
from sklearn.model_selection import train_test_split
#Preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

#Set working directory
os.chdir("C:\\Users\\Santosh Selvaraj\\Documents\\Working Directory\\Data Science Projects\\Telco Customer Churn - ANN")

#Import the dataset
dataset = pd.read_csv("TelcoCustomerChurn.csv", sep = ",")
dataset = dataset[['gender', 'InternetService','Contract', 'PaymentMethod',
         'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
         'PhoneService', 'MultipleLines', 'OnlineSecurity',
         'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
         'StreamingMovies', 'PaperlessBilling', 
         'MonthlyCharges', 'TotalCharges', 'Churn']]

#Creating X and y variables
X = dataset.iloc[:,0:19].values
y = dataset.iloc[:,19].values

#Converting categorical variables into labels


#Split into training and test set
