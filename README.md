## Classification-Project
This is a customer churning prediction project

# Introduction
Customer attrition or churn is a huge expenditure for organizations. Customer churn is the percentage of customers that stopped using a company's product or service within a specific time period. It is a nightmare for any organisation to find out that the number of customers they started the year with have reduced. Actually, the goal is to increase sales by increasing the number of customers purchasing the organisation. if the opposite is happening, it is alarming as it is much more difficult to get new customers. 
To be able to understand this phenomenon, analysis are required to identify the factors influencing customer churn, consistently monitor churn numbers and find ways to minimize this number as customer churn can be hard to fully eradicate.

In this project, we analyzed a telco data to evaluate the possibility of a customer churning, the key indicators of thechurn and some strategies that can be implemented to retain customers.

The CRISP-DM framework was used throughout the analysis

# Hypothesis and Questions

Null hypothesis - Churning ability is influenced by price. Alternate hypothesis - Churning ability is independent of price

Questions

Factors inflencing churning?
Is higher prices attributed to streaming movies?
Which tech support comes with higher payment prices?
Which payment method is the most popular?



# Data cleaning
The data in question had no duplicates and directly visible missing data because the various columns were not in the right datatype. After the datatype conversion, all missing values were replaced with median values

# Exploratory data analysis
From the analysis, higher prices were associated with streaming movies and with subscribing to all phone and internet services. Furthermore, the most popular payment method was electronic checks.
When it comes to churning, factors like contract type, relationship status and age class influences churning. People on month to month contract have a higher chance of churning, so are senior citizens (from the data 41% of senior citizens churned). Also, people with partners churn less.

# Feature engineering
Initially, some columns were feature engineered to reduce the total number of columns. However, after modelling, the evaluation metrics of the model were lower hence in the quest to improve the models, no column was feature engineered to help the modelling process.


# Encoding and scaling
For feature encoding, label and onehot encoders were used. The label encoder was used to encode the dependent variable churn, and the one-hot encoder for the categorical columns. The remaining numerical columns were scaled using the standard scaler to get all variable into the same scale.

# Modelling and evaluation metrics
After data preparation, the processed data was split into train and evaluation set.
After, five models were trained and predicted on the evaluation set. Five evaluation metrics were used to evaluate the various models trained. Unfortunately, our dependent variable had a class imbalance hence our final evaluation metrics were the f1 and fbeta score and not accuracy, precision and recall. Also, because of the class unbalance, the metrics were quite low so the next approach is to balance the data before modelling. However, based on the metrics, the best performing model was the logistic regression classifier.

# Hyperparameter tuning
To confirm our best performing model, some parameters were tuned to improve the performance of the models using the GridCV and searcher. After the hyperparameter tuning, the logistic regressor still performed better.

## Medium article
https://medium.com/@cnorkplim/predicting-customer-churn-rate-using-classification-analysis-4a63e1c1c356

## Author
Linda Adzigbli
