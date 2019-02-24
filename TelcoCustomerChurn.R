#Import necessary libraries
library(ggplot2)
library(tidyverse)
library(caTools)
library(naivebayes)
#Import the dataset
TCN_Dataset = read.csv("TelcoCustomerChurn.csv")

#Exploratory Data Analysis
TCN_Dataset = TCN_Dataset %>%
  mutate_at(vars(SeniorCitizen,Partner,Dependents,
                 #PhoneService,MultipleLines,
                 #OnlineSecurity,OnlineBackup,DeviceProtection,
                 #TechSupport,StreamingTV,StreamingMovies,
                 PaperlessBilling,Churn), as.factor)
#Gender
TCN_Dataset %>% group_by(gender) %>% 
  summarise(Total = sum(as.numeric(Churn))) %>%
  ggplot(aes(gender,Total)) + geom_bar(stat="identity")

#Senior Citizen
TCN_Dataset %>% group_by(SeniorCitizen) %>% 
  summarise(Total = sum(as.numeric(Churn))) %>%
  ggplot(aes(SeniorCitizen,Total)) +
  geom_bar(stat="identity", width = 0.5, fill = "navy blue") +
  labs(title="Churn Rate:Seniors",x="Senior Citizen",y="Churned Customers") +
  theme_classic()
7294/8912

#Partner
TCN_Dataset %>% group_by(Partner) %>% 
  summarise(Total = sum(as.numeric(Churn))) %>%
  ggplot(aes(Partner,Total)) +
  geom_bar(stat="identity", width = 0.5, fill = "navy blue") +
  labs(title="Churn Rate:Customers with Partners",x="Having Partner",y="Churned Customers") +
  theme_classic()
4841/8912
#Dependents
TCN_Dataset %>% group_by(Dependents) %>% 
  summarise(Total = sum(as.numeric(Churn))) %>%
  ggplot(aes(Dependents,Total)) +
  geom_bar(stat="identity", width = 0.5, fill = "navy blue") +
  labs(title="Churn Rate:Customers with Dependents",x="Having Dependents",y="Churned Customers") +
  theme_classic()

#Tenure
TCN_Dataset %>% ggplot(aes(Churn, tenure)) +
  geom_boxplot(width=0.2, color = "navy blue") +
  labs(title="Churn Rate:Customer Tenure",x="Churn",y="Tenure(in Months)") +
  theme_classic()

#Phone Service
TCN_Dataset %>% group_by(PhoneService) %>% 
  summarise(Total = sum(as.numeric(Churn))) %>%
  ggplot(aes(PhoneService,Total)) +
  geom_bar(stat="identity", width = 0.5, fill = "navy blue") +
  labs(title="Churn Rate:Customers with Phone Service",x="Having Phone Service",y="Churned Customers") +
  theme_classic()

#Internet Service
TCN_Dataset %>% group_by(InternetService) %>% 
  summarise(Total = sum(as.numeric(Churn))) %>%
  ggplot(aes(InternetService,Total)) +
  geom_bar(stat="identity", width = 0.3, fill = "navy blue") +
  labs(title="Churn Rate:Customers with Internet Service",x="Having Internet Service",y="Churned Customers") +
  theme_classic()

#Contract
TCN_Dataset %>% group_by(Contract) %>% 
  summarise(Total = sum(as.numeric(Churn))) %>%
  ggplot(aes(Contract,Total)) +
  geom_bar(stat="identity", width = 0.3, fill = "navy blue") +
  labs(title="Churn Rate:Customer Contract",x="Contract",y="Churned Customers") +
  theme_classic()

#Paperless Biling
TCN_Dataset %>% group_by(PaperlessBilling) %>% 
  summarise(Total = sum(as.numeric(Churn))) %>%
  ggplot(aes(PaperlessBilling,Total)) +
  geom_bar(stat="identity", width = 0.3, fill = "navy blue") +
  labs(title="Churn Rate:Paperless Billing",x="Has Paperless Billing",y="Churned Customers") +
  theme_classic()

#Payment Method
TCN_Dataset %>% group_by(PaymentMethod) %>% 
  summarise(Total = sum(as.numeric(Churn))) %>%
  ggplot(aes(PaymentMethod,Total)) +
  geom_bar(stat="identity", width = 0.3, fill = "navy blue") +
  labs(title="Churn Rate:Payment Method",x="Payment Method",y="Churned Customers") +
  theme_classic()

#Total Charges
TCN_Dataset %>% ggplot(aes(Churn, TotalCharges)) +
  geom_boxplot(width=0.2, color = "navy blue") +
  labs(title="Churn Rate:Total Charges",x="Churn",y="TotalCharges") +
  theme_classic()

############################################################################################
#FEATURE SELECTION
#Backward Elimination
model.full = glm(Churn~., data = TCN_Dataset, family = binomial)
model.null = glm(Churn~1, data = TCN_Dataset, family = binomial)
model.final = step(model.full, scope = list(lower = model.null), 
                   direction = "backward")
summary(model.final)

names(TCN_Dataset)
############################################################################################
#DATA PRE-PROCESSING - HYPOTHESIS 1
TCN_Dataset = TCN_Dataset %>%
  mutate(Family = ifelse(Partner==1 | Dependents==1,1,0),
         Internet = ifelse(InternetService != "No",1,0),
         Services = PhoneService + MultipleLines + Internet + OnlineSecurity + OnlineBackup +
           DeviceProtection + TechSupport + StreamingTV + StreamingMovies) %>%
  select(-c(gender,Partner,Dependents,PhoneService,MultipleLines,Internet,OnlineSecurity,
            OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies))

Model.final = glm(Churn~.,data=TCN_Dataset,family = binomial)
summary(Model.final)


#Sample Split
split = sample.split(TCN_Dataset$Churn, SplitRatio = 0.75)
training_set = subset(TCN_Dataset, split == TRUE)
test_set = subset(TCN_Dataset, split == FALSE)

#Scaling
training_set[-3] = scale(training_set[-3])
test_set = scale(test_set)

