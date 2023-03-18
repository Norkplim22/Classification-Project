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
The data in question had no directly visible missing data because the various columns were not in the right datatype. Details of the columns are listed below.
Gender -- Whether the customer is a male or a female
SeniorCitizen -- Whether a customer is a senior citizen or not
Partner -- Whether the customer has a partner or not (Yes, No)
Dependents -- Whether the customer has dependents or not (Yes, No)
Tenure -- Number of months the customer has stayed with the company
Phone Service -- Whether the customer has a phone service or not (Yes, No)
MultipleLines -- Whether the customer has multiple lines or not
InternetService -- Customer's internet service provider (DSL, Fiber Optic, No)
OnlineSecurity -- Whether the customer has online security or not (Yes, No, No Internet)
OnlineBackup -- Whether the customer has online backup or not (Yes, No, No Internet)
DeviceProtection -- Whether the customer has device protection or not (Yes, No, No internet service)
TechSupport -- Whether the customer has tech support or not (Yes, No, No internet)
StreamingTV -- Whether the customer has streaming TV or not (Yes, No, No internet service)
StreamingMovies -- Whether the customer has streaming movies or not (Yes, No, No Internet service)
Contract -- The contract term of the customer (Month-to-Month, One year, Two year)
PaperlessBilling -- Whether the customer has paperless billing or not (Yes, No)
Payment Method -- The customer's payment method (Electronic check, mailed check, Bank transfer(automatic), Credit card(automatic))
MonthlyCharges -- The amount charged to the customer monthly
TotalCharges -- The total amount charged to the customer
Churn -- Whether the customer churned or not (Yes or No)

After the datatype conversion, all missing values were replaced with median values

# Exploratory data analysis

# Feature engineering


# Encoding and scaling

# Modelling and evaluation metrics

# Hyperparameter tuning

## Author
Linda Adzigbli