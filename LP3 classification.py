#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/eaedk/Machine-Learning-Tutorials/blob/main/ML_Step_By_Step_Guide.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Intro
# ## General
# Machine learning allows the user to feed a computer algorithm an immense amount of data and have the computer analyze and make data-driven recommendations and decisions based on only the input data. 
# In most of the situations we want to have a machine learning system to make **predictions**, so we have several categories of machine learning tasks depending on the type of prediction needed: **Classification, Regression, Clustering, Generation**, etc.
# 
# **Classification** is the task whose goal is the prediction of the label of the class to which the input belongs (e.g., Classification of images in two classes: cats and dogs).
# **Regression** is the task whose goal is the prediction of numerical value(s) related to the input (e.g., House rent prediction, Estimated time of arrival ).
# **Generation** is the task whose goal is the creation of something new related to the input (e.g., Text translation, Audio beat generation, Image denoising ). **Clustering** is the task of grouping a set of objects in such a way that objects in the same group (called a **cluster**) are more similar (in some sense) to each other than to those in other **clusters** (e.g., Clients clutering).
# 
# In machine learning, there are learning paradigms that relate to one aspect of the dataset: **the presence of the label to be predicted**. **Supervised Learning** is the paradigm of learning that is applied when the dataset has the label variables to be predicted, known as ` y variables`. **Unsupervised Learning** is the paradigm of learning that is applied when the dataset has not the label variables to be predicted. **Self-supervised Learning** is the paradigm of learning that is applied when part of the X dataset is considere as the label to be predicted (e.g., the Dataset is made of texts and the model try to predict the next word of each sentence).
# 
# ## Notebook overview
# 
# This notebook is a guide to start practicing Machine Learning.

# # Setup

# ## Installation
# Here is the section to install all the packages/libraries that will be needed to tackle the challlenge.

# In[1]:


# !pip install -q <lib_001> <lib_002> ...


# ## Importation
# Here is the section to import all the packages/libraries that will be used through this notebook.

# In[2]:


# Data handling
import pandas as pd
import numpy as np


# Vizualisation (Matplotlib, Plotly, Seaborn, etc. )
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')
from pandas.plotting import lag_plot

# EDA (pandas-profiling, etc. )
import pandas_profiling as pp

# Feature Processing (Scikit-learn processing, etc. )
import sklearn 
import scipy
import statsmodels

# Machine Learning (Scikit-learn Estimators, Catboost, LightGBM, etc. )
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error

# Hyperparameters Fine-tuning (Scikit-learn hp search, cross-validation, etc. )
...

# Other packages
import os

import warnings
warnings.filterwarnings("ignore")


# # Data Loading
# Here is the section to load the datasets (train, eval, test) and the additional files

# In[3]:


# For CSV, use pandas.read_csv
df = pd.read_csv("Telco-Customer-Churn.csv")


# ## Hypothesis and Questions
Hypothesis

Null hypothesis - Churning ability is influenced by price.
Alternate hypothesis - Churning ability is independent of price
# Questions
# 
# 1. What is the rate of churned to unchurned customers 
# 2. How much revenue is lost as a result of churning activities
# 3. Is higher prices attributed to streaming movies
# 4. Which tech support comes with higher payment prices
# 5. Which payment method is the most popular
# 6. Factors inflencing churning
# 
# 

# # Exploratory Data Analysis: EDA
# Here is the section to **inspect** the datasets in depth, **present** it, make **hypotheses** and **think** the *cleaning, processing and features creation*.

# In[4]:


df.shape


# In[5]:


# Code here
df.head()


# In[6]:


pd.set_option('display.max_rows', None)
df


# In[7]:


df_missing = df.isna().sum()
df_missing


# In[8]:


df_duplicate = df.duplicated().sum()
df_duplicate


# In[9]:


df.info()


# In[10]:


df["TotalCharges"].replace(to_replace=" ", value=np.nan, inplace=True)


# In[11]:


df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])


# In[12]:


# handling infinite values
df = df.replace([np.inf, -np.inf], np.nan)


# In[13]:


#Handle missing values
                           
df["TotalCharges"].fillna(value=df["TotalCharges"].median(), inplace=True)                           


# In[14]:


df.info()


# In[15]:


df.isna().sum()


# In[16]:


pd.set_option('display.max_rows', None)
df["TotalCharges"].unique()


# In[17]:


df.describe()


# In[18]:


corr = df.corr()
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')


# In[19]:


#sb.heatmap(df.corr(), annot=True)


# In[20]:


df_profile = pp.ProfileReport(df)
df_profile


# In[21]:


# how to handle class imbalance


# ## What is the rate of churned to unchurned customers?

# In[22]:


# count
Churned_customers = len(df[df["Churn"] == "Yes"])
Unchurned_customers = len(df[df["Churn"] == "No"])


# In[23]:


# percentage
churn = (Churned_customers/len(df))*100
unchurn = 100-churn


# In[24]:


#visuals

labels = ["Churned", "Unchurned"]
sizes = [churn, unchurn]
colors = ["red", "blue"]
explode = (0.1, 0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct="%1.1f%%", shadow=True, startangle=90)
plt.axis("equal")
plt.title("Percentage of Churned to Unchurned Customers")
plt.show()

counts = (Churned_customers, Unchurned_customers)
plt.bar(labels, counts, color=colors)
plt.title("Number of Churned to Unchurned customers")
plt.xlabel("Churn status")
plt.ylabel("Count")
plt.show()


# ## How much revenue is lost as result of churning activities?
The total revenue lost monthly is $139130.85.
With the right strategies, the company can save about $1669570.2 yearly
# In[26]:


#Monthly and Yearly revenue

Total_month_revenue = round(df["MonthlyCharges"].sum(), 2)
Yearly_revenue = round(Total_month_revenue * 12, 2)

Churned_revenue = df.loc[df["Churn"] == "Yes", "MonthlyCharges"].sum()
actual_month_rev = Total_month_revenue - Churned_revenue
yearly_chun_rev = round(Churned_revenue * 12, 2)

print(f"Total revenue lost monthly is ${Churned_revenue}")
print(f"Total revenue lost yearly is ${yearly_chun_rev}")


# In[28]:


# Visuals

labels = ["Monthly Revenue", "Lost Monthly Revenue", "Actual Monthly Revenue", "Potential Yearly Savings", "Potential Yearly Revenue"]
values = [Total_month_revenue, Churned_revenue, actual_month_rev, yearly_chun_rev, Yearly_revenue]

# Set seaborn style
sns.set_style("whitegrid")
plt.figure(figsize=(15,7))

# Create the bar chart
ax = sns.barplot(x=labels, y=values)
ax.bar_label(ax.containers[0], label_type='edge')
ax.set_title("Comprehensive Analysis of Revenue with and without Churn activities")
plt.tick_params(axis='x', labelrotation=45)
ax.set_xlabel("Revenue")
ax.set_ylabel("Amount($)")


# In[ ]:





# ## Is higher prices attributed to streaming movies? YES

# In[29]:


df_movies = df[["StreamingMovies", "TotalCharges"]]
df_movies.sort_values(by="TotalCharges", ascending=False)


# In[ ]:





# ## Which services comes with higher payment prices? 
# All phone and internet services

# In[30]:


Services = df[["StreamingMovies", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "Churn", "TotalCharges"]]
Services.sort_values(by="TotalCharges", ascending=False, inplace=True)


# In[31]:


Services.head(20)


# ## Which payment method is the most popular?
# Electronic check

# In[32]:


payment_method = df.groupby("PaymentMethod")["TotalCharges"].count() 
payment_method.plot()
plt.tick_params(axis='x', labelrotation=45)


# ## Factors influencing churning
People on month to month contract have a higher chance of churning.
Senior citizens have higher chance of churning (from the data 41% of senior citizens churned)
People with partners churn less.
People without dependents are more likely to churn
Churning is independent of the gender type
# In[33]:


sns.set(style="whitegrid")
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20,8))

# Factor 1, Senior citizen
sns.countplot(x="SeniorCitizen", hue="Churn", data=df, ax = axs[0,0])

#Factor 2, Relatonship status
sns.countplot(x="Partner", hue="Churn", data=df, ax = axs[0,1])
          
#Factor 3, having dependents
sns.countplot(x="Dependents", hue="Churn", data=df, ax = axs[0,2])
          
#Factor 4, Gender
sns.countplot(x="gender", hue="Churn", data=df, ax = axs[1,0])
          
#Factor 5, Contract type
sns.countplot(x="Contract", hue="Churn", data=df, ax=axs[1,1])

#Factor 5, Internet type
sns.countplot(x="InternetService", hue="Churn", data=df, ax=axs[1,2])

fig.tight_layout()
plt.show()


# In[ ]:





# # Feature Processing & Engineering
# Here is the section to **clean**, **process** the dataset and **create new features**.

# ## Drop Duplicates

# In[34]:


# Use pandas.DataFrame.drop_duplicates method

print("There are no duplicates in the data") if df_duplicate == 0 else print(f"'{df_duplicate}' duplicate rows in the data")


# ## Impute Missing Values

# In[ ]:


# imputed already


# ## Check class imbalance

# In[35]:


# class inbalance on churn column
colors = ["blue", "red",]
labels = ['No', 'Yes']
sns.set_palette(sns.color_palette(colors))
sns.countplot(df.Churn)
plt.figure(figsize=(10,8))


# ## New Features Creation

# In[ ]:


# Code here

#df["PhoneService"] = df.apply(lambda x: "MultipleLines" if x["MultipleLines"] == 'Yes' and x["PhoneService"] =="Yes" 
                             # else "SingleLine" if x["MultipleLines"] == "No" and x["PhoneService"] =="Yes"
                             # else "None", axis=1)
                               
#df["StreamingService"] = df.apply(lambda x: "Fullservice" if x["StreamingTV"] == "Yes" and x["StreamingMovies"] == "Yes"
                                 #else "None" if x["StreamingTV"] == "No" and x["StreamingMovies"] == "No"
                                # else "TV" if x["StreamingTV"] == "Yes" and x["StreamingMovies"] == "No"
                               #  else "Movies" if x["StreamingTV"] == "No" and x["StreamingMovies"] == "Yes" 
                                # else "NoInternet", axis=1)

#df["SecurityService"] = df.apply(lambda x: "Fullsecurity" if x["OnlineSecurity"] == "Yes" and x["DeviceProtection"] == "Yes"
                                # else "None" if x["OnlineSecurity"] == "No" and x["DeviceProtection"] == "No"
                                # else "Online" if x["OnlineSecurity"] == "Yes" and x["DeviceProtection"] == "No"
                                # else "Device" if x["OnlineSecurity"] == "No" and x["DeviceProtection"] == "Yes" 
                               #  else "NoInternet", axis=1)


# In[36]:


df.head()


# ## Features Encoding
# 
# 
# 

# In[37]:


# Convert columns to binary integers

df["Churn"].replace({"No":0, "Yes":1}, inplace=True)


# In[38]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df["Churn"] = label_encoder.fit_transform(df["Churn"])


# In[39]:


# From sklearn.preprocessing use OneHotEncoder to encode the categorical features.

from sklearn.preprocessing import OneHotEncoder

cat_col = df[["Contract", "PaymentMethod", "gender", "Partner", "Dependents", "InternetService", "TechSupport", "OnlineBackup", "PhoneService","MultipleLines", "StreamingTV", "StreamingMovies", "OnlineSecurity", "DeviceProtection"]]

encoder = OneHotEncoder (drop = "first", sparse=False)
encoded_features = encoder.fit_transform(cat_col)

encoded_cat_col = pd.DataFrame(encoded_features, columns=encoder.get_feature_names())


# In[40]:


encoded_cat_col.head()


# In[41]:


df.info()


# In[42]:


encoded_cat_col.info()


# In[43]:


# modify data by dropping columns used for encoding and other columns not necessary for modeling
# data with only numeric columns

mod_df = df.drop(["customerID", "PaperlessBilling", "Contract", "PaymentMethod", "gender", "Partner", "Dependents", "InternetService", "TechSupport", "OnlineBackup", "PhoneService","MultipleLines", "StreamingTV", "StreamingMovies", "OnlineSecurity", "DeviceProtection"], axis=1)
mod_df.head() 


# In[44]:


# merge all data  
new_df = pd.concat([mod_df, encoded_cat_col], axis=1)
new_df.head()


# In[45]:


new_df.info()


# In[ ]:





# ## Features Scaling
# 

# In[46]:


# scale data without encoded categorical features

train_num_columns = new_df[["tenure", "MonthlyCharges", "TotalCharges"]]


# In[47]:


from sklearn.preprocessing import StandardScaler

scaled_data = StandardScaler().fit_transform(train_num_columns)
scaled_mod_df = pd.DataFrame(data = scaled_data)
scaled_mod_df.columns = train_num_columns.columns.values


# In[48]:


scaled_mod_df


# In[49]:


new_df1 = new_df.drop(["tenure", "MonthlyCharges", "TotalCharges"], axis=1)
new_df1.info()


# In[50]:


scaled_train_fulldf = pd.concat([new_df1, scaled_mod_df], axis=1)
scaled_train_fulldf.info()


# ## Dataset Splitting

# In[51]:


# split train data into train and evaluation set

X = scaled_train_fulldf.drop(["Churn"], axis =1, inplace=False)
y = scaled_train_fulldf["Churn"]

from sklearn.model_selection import train_test_split

X_train, X_eval, y_train, y_eval = train_test_split(X,y, test_size = 0.2, random_state = 42)


# In[52]:


X_train.shape, X_eval.shape, y_train.shape, y_eval.shape


# ## Optional: Train Dataset Balancing 

# In[ ]:


# Use Over-sampling/Under-sampling methods, more details here: https://imbalanced-learn.org/stable/install.html

#from imblearn.over_sampling import SMOTE
#smote = SMOTE(random_state=27, sampling_strategy=1.0)
#T_train_bal, V_train_bal = sm.fit_resample(T_train, V_train)


# # Machine Learning Modeling 
# Here is the section to **build**, **train**, **evaluate** and **compare** the models to each others.

# ## Logistic Regression
# 
# Please, keep the following structure to try all the model you want.

# ### Create and train the Model

# In[ ]:


#np.any(np.isnan(train_df))


# In[ ]:


#np.all(np.isfinite(train_df))


# In[53]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score

logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# ### Predict and evaluate the Model on the Evaluation dataset (Evalset)

# In[54]:


# Compute the valid metrics for the use case # Optional: show the classification report
from sklearn.metrics import f1_score

# Predicting the Test set results
log_pred  = logreg.predict(X_eval)

#Evaluate results
acc_score = accuracy_score(y_eval, log_pred)
pre_score = precision_score(y_eval, log_pred)
rec_score = recall_score(y_eval, log_pred )
f1_score = f1_score(y_eval, log_pred )
f2_score = fbeta_score(y_eval, log_pred, beta=2.0)

df_model_result = pd.DataFrame([["Logistic Regression", acc_score, pre_score, rec_score, f1_score, f2_score]], columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "F2 Score"])
df_model_result


# In[55]:


#confusion matrix
from sklearn.metrics import confusion_matrix
logreg_cm = confusion_matrix(y_eval, log_pred)

# Plot the confusion matrix as a heatmap using Seaborn
sns.heatmap(logreg_cm, annot=True, linewidth=0.5, fmt=".0f", cmap='RdPu')

plt.title("Logistion regressor Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# ## K Nearest Neighbour 

# ### Create and train the Model

# In[56]:


# Code here
from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier()
KNN.fit(X_train, y_train)


# ### Predict and evaluate the Model on the Evaluation dataset (Evalset)

# In[57]:


# Predicting the Test set results
KNN_pred  = KNN.predict(X_eval)

#Evaluate results
acc_score = accuracy_score(y_eval, KNN_pred)
pre_score = precision_score(y_eval, KNN_pred)
rec_score = recall_score(y_eval, KNN_pred)
f1_score = sklearn.metrics.f1_score(y_eval, KNN_pred)
f2_score = fbeta_score(y_eval, KNN_pred, beta=2.0)

model_results = pd.DataFrame([["KNN", acc_score, pre_score, rec_score, f1_score, f2_score]], columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "F2 Score"])
df_model_results = df_model_result.append(model_results, ignore_index = True)
df_model_results


# In[58]:


#confusion matrix
from sklearn.metrics import confusion_matrix
KNN_cm = confusion_matrix(y_eval, KNN_pred)

# Plot the confusion matrix as a heatmap using Seaborn
sns.heatmap(KNN_cm, annot=True, linewidth=0.5, fmt=".0f", cmap='RdPu')

plt.title("K Nearest Neighbors Classifier Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# ## Decision Tree

# ### Create and train the Model

# In[59]:


from sklearn.tree import DecisionTreeClassifier
# Fitting Decision Tree to the Training set

dec_tree = DecisionTreeClassifier()
dec_tree.fit(X_train, y_train)


# ### Predict and evaluate the Model on the Evaluation dataset (Evalset)

# In[60]:


# Predicting the Test set results
tree_pred  = dec_tree.predict(X_eval)

#Evaluate results
acc_score = accuracy_score(y_eval, tree_pred)
pre_score = precision_score(y_eval, tree_pred)
rec_score = recall_score(y_eval, tree_pred)
f1_score = sklearn.metrics.f1_score(y_eval, tree_pred)
f2_score = fbeta_score(y_eval, tree_pred, beta=2.0)

model_results = pd.DataFrame([["Decision Tree", acc_score, pre_score, rec_score, f1_score, f2_score]], columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "F2 Score"])
df_models_results = df_model_results.append(model_results, ignore_index = True)
df_models_results


# In[61]:


#confusion matrix
from sklearn.metrics import confusion_matrix
dt_cm = confusion_matrix(y_eval, tree_pred)

# Plot the confusion matrix as a heatmap using Seaborn
sns.heatmap(dt_cm, annot=True, linewidth=0.5, fmt=".0f", cmap='RdPu')

plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# ## Xg Boost

# ### Create and train the Model

# In[62]:


from sklearn.ensemble import GradientBoostingClassifier

xgc = GradientBoostingClassifier() 
xgc.fit(X_train, y_train)


# ### Predict and evaluate the Model on the Evaluation dataset (Evalset)

# In[63]:


# Predicting the Test set results
xgc_pred  = xgc.predict(X_eval)

#Evaluate results
acc_score = accuracy_score(y_eval, xgc_pred)
pre_score = precision_score(y_eval, xgc_pred)
rec_score = recall_score(y_eval, xgc_pred)
f1_score = sklearn.metrics.f1_score(y_eval, xgc_pred)
f2_score = fbeta_score(y_eval, xgc_pred, beta=2.0)

model_results = pd.DataFrame([["XGB Classifier", acc_score, pre_score, rec_score, f1_score, f2_score]], columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "F2 Score"])
df_models_results = df_models_results.append(model_results, ignore_index = True)
df_models_results


# In[64]:


#confusion matrix
from sklearn.metrics import confusion_matrix
xgc_cm = confusion_matrix(y_eval, xgc_pred)

# Plot the confusion matrix as a heatmap using Seaborn
sns.heatmap(xgc_cm, annot=True, linewidth=0.5, fmt=".0f", cmap='RdPu')

plt.title("Xg Boost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# ## Random Forest Classifier

# ### Create and train the Model

# In[65]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)


# ### Predict and evaluate the Model on the Evaluation dataset (Evalset)

# In[66]:


# Predicting the Test set results
rfc_pred  = rfc.predict(X_eval)

#Evaluate results
acc_score = accuracy_score(y_eval, rfc_pred)
pre_score = precision_score(y_eval, rfc_pred)
rec_score = recall_score(y_eval, rfc_pred)
f1_score = sklearn.metrics.f1_score(y_eval, rfc_pred)
f2_score = fbeta_score(y_eval, rfc_pred, beta=2.0)

model_results = pd.DataFrame([["Random Forest", acc_score, pre_score, rec_score, f1_score, f2_score]], columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "F2 Score"])
df_models_results = df_models_results.append(model_results, ignore_index = True)
df_models_results


# In[67]:


#confusion matrix
from sklearn.metrics import confusion_matrix
rfc_cm = confusion_matrix(y_eval, rfc_pred)

# Plot the confusion matrix as a heatmap using Seaborn
sns.heatmap(rfc_cm, annot=True, linewidth=0.5, fmt=".0f", cmap='RdPu')

plt.title("Random Forest Classifier Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# ## Models comparison
# Create a pandas dataframe that will allow you to compare your models.
# 
# Find a sample frame below :
# 
# |     | Model_Name     | Metric (metric_name)    | Details  |
# |:---:|:--------------:|:--------------:|:-----------------:|
# | 0   |  -             |  -             | -                 |
# | 1   |  -             |  -             | -                 |
# 
# 
# You might use the pandas dataframe method `.sort_values()` to sort the dataframe regarding the metric.

# In[69]:


model_select = df_models_results[["Model", "F1 Score"]].sort_values(by="F1 Score", ascending=False)
model_select                               


# ## Hyperparameters tuning 
# 
# Fine-tune the Top-k models (3 < k < 5) using a ` GridSearchCV`  (that is in sklearn.model_selection
# ) to find the best hyperparameters and achieve the maximum performance of each of the Top-k models, then compare them again to select the best one.

# ### Hypertune best model 1

# In[70]:


# Code here
logreg.get_params()


# In[71]:


parameters = {
  'C':[2, 10],
  'solver' : ['newton-cg', 'newton-cholesky', 'sag'],  
  'max_iter': [120, 200, 250], 
  'class_weight': ['balanced'],
  'random_state': [10, 25, 40],
  'multi_class': ['auto', 'ovr', 'multinomial'],
}


# In[72]:


# instantiate the Searcher
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score

searcher = GridSearchCV(
    estimator=logreg,
    param_grid=parameters,
    scoring=["accuracy", "precision", "recall", "f1"], 
    refit="f1", 
    cv=5,  
    verbose=3 
)


# In[73]:


searcher.fit(X_train, y_train)


# In[74]:


search_hist = pd.DataFrame(searcher.cv_results_)
search_hist


# In[75]:


searcher.best_params_


# In[76]:


searcher.best_estimator_


# In[77]:


logreg1 = searcher.best_estimator_
logreg1.fit(X_train, y_train)

log1_pred  = logreg1.predict(X_eval)

#Evaluate results
acc_score = accuracy_score(y_eval, log1_pred)
pre_score = precision_score(y_eval, log1_pred)
rec_score = recall_score(y_eval, log1_pred )
f1_score = sklearn.metrics.f1_score(y_eval, log1_pred )
f2_score = fbeta_score(y_eval, log1_pred, beta=2.0)

model_results = pd.DataFrame([["Logistic Regression1", acc_score, pre_score, rec_score, f1_score, f2_score]], columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "F2 Score"])
df_models_results = df_models_results.append(model_results, ignore_index = True)
df_models_results


# ### Hypertune best model 2 (KNN)

# In[78]:


KNN.get_params


# In[79]:


KNN.get_params().keys()


# In[80]:


parameters = {
    'n_neighbors': [2, 5, 7],
    'algorithm': ['ball_tree', 'kd_tree', 'brute'], 
    'leaf_size': [10, 15, 20],
}


# In[81]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score

search = GridSearchCV(
    estimator=KNN,
    param_grid=parameters,
    scoring=["accuracy", "precision", "recall", "f1"], 
    refit="f1", 
    cv=5,  
    verbose=3 
)


# In[82]:


search.fit(X_train, y_train)


# In[83]:


search_hist1 = pd.DataFrame(search.cv_results_)
search_hist1


# In[84]:


search.best_params_


# In[85]:


search.best_estimator_


# In[86]:


KNN1 = search.best_estimator_
KNN1.fit(X_train, y_train)

KNN1_pred  = KNN1.predict(X_eval)

#Evaluate results
acc_score = accuracy_score(y_eval, KNN1_pred)
pre_score = precision_score(y_eval, KNN1_pred)
rec_score = recall_score(y_eval, KNN1_pred )
f1_score = sklearn.metrics.f1_score(y_eval, KNN1_pred )
f2_score = fbeta_score(y_eval, KNN1_pred, beta=2.0)

model_results = pd.DataFrame([["KNN1", acc_score, pre_score, rec_score, f1_score, f2_score]], columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "F2 Score"])
df_models_results = df_models_results.append(model_results, ignore_index = True)
df_models_results


# In[ ]:





# ### Hypertune best model 3 (XGB Classifier)

# In[87]:


xgc.get_params()


# In[88]:


parameters = {
    'n_estimators': [100, 200],
    'random_state': [10, 20],
    'max_depth': [4, 6],
    'max_features': [2, 5],
    'min_impurity_decrease': [1.0], 
    'min_samples_leaf': [4, 5],
    'min_samples_split': [2, 5],
}


# In[89]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score

search = GridSearchCV(
    estimator=xgc,
    param_grid=parameters,
    scoring=["accuracy", "precision", "recall", "f1"], 
    refit="f1", 
    cv=5,  
    verbose=3 
)


# In[90]:


search.fit(X_train, y_train)


# In[91]:


search_hist1 = pd.DataFrame(search.cv_results_)
search_hist1


# In[92]:


search.best_params_


# In[93]:


search.best_estimator_


# In[94]:


xgc1 = search.best_estimator_
xgc1.fit(X_train, y_train)

xgc1_pred  = xgc1.predict(X_eval)

#Evaluate results
acc_score = accuracy_score(y_eval, xgc1_pred)
pre_score = precision_score(y_eval, xgc1_pred)
rec_score = recall_score(y_eval, xgc1_pred )
f1_score = sklearn.metrics.f1_score(y_eval, xgc1_pred )
f2_score = fbeta_score(y_eval, xgc1_pred, beta=2.0)

model_results = pd.DataFrame([["XG BOOST1", acc_score, pre_score, rec_score, f1_score, f2_score]], columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "F2 Score"])
df_models_results = df_models_results.append(model_results, ignore_index = True)
df_models_results


# ### Hypertune best model 4 (Random Forest Classifier)

# In[95]:


rfc.get_params()


# In[96]:



parameters = {
    'criterion': ["gini","entropy"],
    'random_state': [10],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 5],
    'n_estimators': [200, 300, 400],
}
     


# In[97]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score

searcher2 = GridSearchCV(
    estimator=rfc,
    param_grid=parameters,
    scoring=["accuracy", "precision", "recall", "f1"], 
    refit="f1", 
    cv=5,  
    verbose=3 
)


# In[98]:


searcher2.fit(X_train, y_train)


# In[99]:


searcher2_hist1 = pd.DataFrame(searcher2.cv_results_)
searcher2_hist1


# In[100]:


searcher2.best_params_


# In[101]:


searcher2.best_estimator_


# In[102]:


rfc1 = searcher2.best_estimator_
rfc1.fit(X_train, y_train)

rfc1_pred  = rfc1.predict(X_eval)

#Evaluate results
acc_score = accuracy_score(y_eval, rfc1_pred)
pre_score = precision_score(y_eval, rfc1_pred)
rec_score = recall_score(y_eval, rfc1_pred )
f1_score = sklearn.metrics.f1_score(y_eval, rfc1_pred )
f2_score = fbeta_score(y_eval, rfc1_pred, beta=2.0)

model_results = pd.DataFrame([["RANDOM FOREST1", acc_score, pre_score, rec_score, f1_score, f2_score]], columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "F2 Score"])
df_models_results = df_models_results.append(model_results, ignore_index = True)
df_models_results


# In[115]:


models = df_models_results[["Model", "F1 Score"]].sort_values(by="F1 Score", ascending=False)
models.head(5)


# # Export key components
# Here is the section to **export** the important ML objects that will be use to develop an app: *Encoder, Scaler, ColumnTransformer, Model, Pipeline, etc*.

# In[116]:


# Use pickle : put all your key components in a python dictionary and save it as a file that will be loaded in an app

#import best model with pickle

import pickle

with open("logreg_model.pkl", "wb") as f:
    pickle.dump(log1_pred, f)


# In[ ]:


#load model

#with open("logreg_model.pkl", "rb") as f:
    #model = pickle.load(f)

