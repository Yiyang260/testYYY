# 01-BU

"""

Explain the business case for data mining

"""

# 02-DU
# Load Dataset
oxid_co2_excel='owid-co2.xlsx'
oxid_co2_excel_2='owid-co2(1).xlsx'

import pandas as pd
import numpy as np
oxid_co2_1=pd.read_excel(oxid_co2_excel)
oxid_co2_1.info() #showinng basic info
oxid_co2_1.head()#showing the first five obsveration
oxid_co2_1.tail()#showing the last five obsveration

oxid_co2_2=pd.read_excel(oxid_co2_excel_2)

# Explore dataset, summary statistics

oxid_summary_stat=oxid_co2_1.describe() #summary stats for all variables
oxid_summary_stat=oxid_co2_1.describe().transpose() #rows become columns and columns become rows
oxid_summary_stat=round(oxid_co2_1.describe().transpose(),2) #round up the numbers into 2 decimal place

co2_summary_stat=oxid_co2_1['co2'].describe()

group_summary_stat=oxid_co2_1[['co2','population','gdp','methane','nitrous_oxide']].describe()

correlation_1=round(oxid_co2_1[['co2','population','gdp','methane','nitrous_oxide']].corr(),2) #find correlations among all attributes

# Visulations
oxid_co2_1['co2'].plot()

oxid_co2_1['co2'].plot.hist(bins=5, color='red', xlabel='co2 range', title='Histogram of co2') #not good, ew

oxid_co2_1['co2'].plot.density()

oxid_co2_1['co2'].plot.box()

oxid_co2_1['state'].value_counts().plot.bar()

oxid_co2_1.plot.box(column='co2', by='state')

oxid_co2_1.plot.scatter(x='year', y='co2')

# Quality check
oxid_co2_1.isnull().sum()

oxid_co2_1.shape

# 03-DP
# Add any pre-processing steps
group_summary_stat2=round(oxid_co2_1[['co2','co2_per_capita']].describe(),2) #round up the numbers into 2 decimal place
selected_data=oxid_co2_1[['country','state','year','co2','methane','nitrous_oxide','population','gdp']] #selected variables that related to objective

# Impute missing value
mean_value_co2 = round(selected_data['co2'].mean(),2)
selected_data['co2']=np.where(selected_data['co2'].isnull(), mean_value_co2, selected_data['co2']) #substitute all null values to the mean value

# Repeat previous step for other attributes
mean_value_methane = round(selected_data['methane'].mean(),2)
selected_data['methane']=np.where(selected_data['methane'].isnull(), mean_value_methane, selected_data['methane'])

mean_value_nitrous_oxide = round(selected_data['nitrous_oxide'].mean(),2)
selected_data['nitrous_oxide']=np.where(selected_data['nitrous_oxide'].isnull(), mean_value_nitrous_oxide, selected_data['nitrous_oxide'])

mean_value_population = round(selected_data['population'].mean(),2)
selected_data['population']=np.where(selected_data['population'].isnull(), mean_value_population, selected_data['population'])

mean_value_gdp = round(selected_data['gdp'].mean(),2)
selected_data['gdp']=np.where(selected_data['gdp'].isnull(), mean_value_gdp, selected_data['gdp'])

selected_data['co2'].plot()

selected_data['methane'].plot()

selected_data['nitrous_oxide'].plot()

selected_data['population'].plot()

selected_data['gdp'].plot()

# Handle extremes and outliers
Q1_co2 = np.percentile(selected_data['co2'], 25, interpolation='midpoint')
Q3_co2 = np.percentile(selected_data['co2'], 75, interpolation='midpoint')
IQR = Q3_co2 - Q1_co2 #Find IQR
upper_co2=Q3_co2+1.5*IQR #Find upper interquartile range
selected_data['co2']=np.where(selected_data['co2'] > upper_co2, np.nan, selected_data['co2']) #elimate values that greater than upper interquartile
mean_value_co2_2 = round(selected_data['co2'].mean(),2)
selected_data['co2']=np.where(selected_data['co2'].isnull(), mean_value_co2_2, selected_data['co2'])

# Repeat previous step for other attributes
Q1_methane = np.percentile(selected_data['methane'], 25, interpolation='midpoint')
Q3_methane = np.percentile(selected_data['methane'], 75, interpolation='midpoint')
IQR2 = Q3_methane - Q1_methane
upper_methane=Q3_methane+1.5*IQR2
selected_data['methane']=np.where(selected_data['methane'] > upper_methane, np.nan, selected_data['methane'])
mean_value_methane_2 = round(selected_data['methane'].mean(),2)
selected_data['methane']=np.where(selected_data['methane'].isnull(), mean_value_methane_2, selected_data['methane'])

Q1_nitrous_oxide = np.percentile(selected_data['nitrous_oxide'], 25, interpolation='midpoint')
Q3_nitrous_oxide = np.percentile(selected_data['nitrous_oxide'], 75, interpolation='midpoint')
IQR3 = Q3_nitrous_oxide - Q1_nitrous_oxide
upper_nitrous_oxide=Q3_nitrous_oxide+1.5*IQR3
selected_data['nitrous_oxide']=np.where(selected_data['nitrous_oxide'] > upper_nitrous_oxide, np.nan, selected_data['nitrous_oxide'])
mean_value_nitrous_oxide_2 = round(selected_data['nitrous_oxide'].mean(),2)
selected_data['nitrous_oxide']=np.where(selected_data['nitrous_oxide'].isnull(), mean_value_nitrous_oxide_2, selected_data['nitrous_oxide'])

Q1_population = np.percentile(selected_data['population'], 25, interpolation='midpoint')
Q3_population = np.percentile(selected_data['population'], 75, interpolation='midpoint')
IQR4 = Q3_population - Q1_population
upper_population=Q3_population+1.5*IQR4
selected_data['population']=np.where(selected_data['population'] > upper_population, np.nan, selected_data['population'])
mean_value_population_2 = round(selected_data['population'].mean(),2)
selected_data['population']=np.where(selected_data['population'].isnull(), mean_value_population_2, selected_data['population'])

Q1_gdp = np.percentile(selected_data['gdp'], 25, interpolation='midpoint')
Q3_gdp = np.percentile(selected_data['gdp'], 75, interpolation='midpoint')
IQR5 = Q3_gdp - Q1_gdp
upper_gdp=Q3_gdp+1.5*IQR5
selected_data['gdp']=np.where(selected_data['gdp'] > upper_gdp, np.nan, selected_data['gdp'])
mean_value_gdp_2 = round(selected_data['gdp'].mean(),2)
selected_data['gdp']=np.where(selected_data['gdp'].isnull(), mean_value_gdp_2, selected_data['gdp'])

selected_data.isnull().sum() #test if null value still exists

selected_data['co2'].plot.box() #show the box plot for processed attribute - co2

selected_data_summary=selected_data[['co2','population','gdp','methane','nitrous_oxide']].describe()

# Create new dummy variable
condition1 = selected_data['co2'] <= 30 # level1
selected_data['co2_level']=np.where(condition1, 'Low CO2', 'High CO2') #create new dymmy variable
selected_data['co2_level'].describe()

selected_data['co2_level'].value_counts().plot.bar()

selected_data['co2_level'].value_counts().plot.pie()

# Integrate two datasets
integrated_co2=selected_data.merge(oxid_co2_2)

# Format the data as required
selected_data2=integrated_co2[['country','state','year','co2','co2_level','methane','nitrous_oxide','population','gdp','coal_co2','cement_co2','flaring_co2','gas_co2','oil_co2','primary_energy_consumption']] #selected variables that related to objective

correlation_2=round(selected_data2[['co2','year','population','gdp','methane','nitrous_oxide','primary_energy_consumption']].corr(),2)
correlation_3=round(selected_data2[['co2','coal_co2','flaring_co2','cement_co2','gas_co2','oil_co2']].corr(),2)

# Impute missing value
mean_value_coal = round(selected_data2['coal_co2'].mean(),2)
selected_data2['coal_co2']=np.where(selected_data2['coal_co2'].isnull(), mean_value_coal, selected_data2['coal_co2'])

mean_value_cement = round(selected_data2['cement_co2'].mean(),2)
selected_data2['cement_co2']=np.where(selected_data2['cement_co2'].isnull(), mean_value_cement, selected_data2['cement_co2'])

mean_value_flaring = round(selected_data2['flaring_co2'].mean(),2)
selected_data2['flaring_co2']=np.where(selected_data2['flaring_co2'].isnull(), mean_value_flaring, selected_data2['flaring_co2'])

mean_value_gas = round(selected_data2['gas_co2'].mean(),2)
selected_data2['gas_co2']=np.where(selected_data2['gas_co2'].isnull(), mean_value_gas, selected_data2['gas_co2'])

mean_value_oil = round(selected_data2['oil_co2'].mean(),2)
selected_data2['oil_co2']=np.where(selected_data2['oil_co2'].isnull(), mean_value_oil, selected_data2['oil_co2'])

mean_value_energy = round(selected_data2['primary_energy_consumption'].mean(),2)
selected_data2['primary_energy_consumption']=np.where(selected_data2['primary_energy_consumption'].isnull(), mean_value_energy, selected_data2['primary_energy_consumption'])

# Handle extremes and outliers
Q1_coal = np.percentile(selected_data2['coal_co2'], 25, interpolation='midpoint') 
Q3_coal = np.percentile(selected_data2['coal_co2'], 75, interpolation='midpoint')
IQR6 = Q3_coal - Q1_coal
upper_coal=Q3_coal+1.5*IQR6
selected_data2['coal_co2']=np.where(selected_data2['coal_co2'] > upper_co2, np.nan, selected_data2['coal_co2'])
mean_value_coal_2 = round(selected_data2['coal_co2'].mean(),2)
selected_data2['coal_co2']=np.where(selected_data2['coal_co2'].isnull(), mean_value_coal_2, selected_data2['coal_co2'])

Q1_cement = np.percentile(selected_data2['cement_co2'], 25, interpolation='midpoint')
Q3_cement = np.percentile(selected_data2['cement_co2'], 75, interpolation='midpoint')
IQR7 = Q3_cement - Q1_cement
upper_cement=Q3_coal+1.5*IQR7
selected_data2['cement_co2']=np.where(selected_data2['cement_co2'] > upper_cement, np.nan, selected_data2['cement_co2'])
mean_value_cement_2 = round(selected_data2['cement_co2'].mean(),2)
selected_data2['cement_co2']=np.where(selected_data2['cement_co2'].isnull(), mean_value_cement_2, selected_data2['cement_co2'])

Q1_flaring = np.percentile(selected_data2['flaring_co2'], 25, interpolation='midpoint')
Q3_flaring = np.percentile(selected_data2['flaring_co2'], 75, interpolation='midpoint')
IQR8 = Q3_flaring - Q1_flaring
upper_flaring=Q3_flaring+1.5*IQR8
selected_data2['flaring_co2']=np.where(selected_data2['flaring_co2'] > upper_flaring, np.nan, selected_data2['flaring_co2'])
mean_value_flaring_2 = round(selected_data2['flaring_co2'].mean(),2)
selected_data2['flaring_co2']=np.where(selected_data2['flaring_co2'].isnull(), mean_value_flaring_2, selected_data2['flaring_co2'])

Q1_gas = np.percentile(selected_data2['gas_co2'], 25, interpolation='midpoint')
Q3_gas = np.percentile(selected_data2['gas_co2'], 75, interpolation='midpoint')
IQR9 = Q3_gas - Q1_gas
upper_gas=Q3_gas+1.5*IQR9
selected_data2['gas_co2']=np.where(selected_data2['gas_co2'] > upper_gas, np.nan, selected_data2['gas_co2'])
mean_value_gas_2 = round(selected_data2['gas_co2'].mean(),2)
selected_data2['gas_co2']=np.where(selected_data2['gas_co2'].isnull(), mean_value_gas_2, selected_data2['gas_co2'])

Q1_oil = np.percentile(selected_data2['oil_co2'], 25, interpolation='midpoint')
Q3_oil = np.percentile(selected_data2['oil_co2'], 75, interpolation='midpoint')
IQR10 = Q3_oil - Q1_oil
upper_oil=Q3_oil+1.5*IQR10
selected_data2['oil_co2']=np.where(selected_data2['oil_co2'] > upper_oil, np.nan, selected_data2['oil_co2'])
mean_value_oil_2 = round(selected_data2['oil_co2'].mean(),2)
selected_data2['oil_co2']=np.where(selected_data2['oil_co2'].isnull(), mean_value_oil_2, selected_data2['oil_co2'])

Q1_energy = np.percentile(selected_data2['primary_energy_consumption'], 25, interpolation='midpoint')
Q3_energy = np.percentile(selected_data2['primary_energy_consumption'], 75, interpolation='midpoint')
IQR11 = Q3_energy - Q1_energy
upper_energy=Q3_energy+1.5*IQR11
selected_data2['primary_energy_consumption']=np.where(selected_data2['primary_energy_consumption'] > upper_energy, np.nan, selected_data2['primary_energy_consumption'])
mean_value_energy_2 = round(selected_data2['primary_energy_consumption'].mean(),2)
selected_data2['primary_energy_consumption']=np.where(selected_data2['primary_energy_consumption'].isnull(), mean_value_energy_2, selected_data2['primary_energy_consumption'])

selected_data2.isnull().sum() #test if null value still exists

selected_data2_summary=selected_data2[['co2','methane','nitrous_oxide','population','gdp','coal_co2','cement_co2','flaring_co2','gas_co2','oil_co2','primary_energy_consumption']].describe()

# 04-DT
# Add any transformation steps
# Feature selection

selected_data2['population']=selected_data2['population']/100000 #convert units to 100000
selected_data2['gdp']=selected_data2['gdp']/100000 #convert units to 100000

from sklearn.feature_selection import SelectKBest, f_classif

# For objective1
Y=selected_data2['co2']
X=selected_data2.drop(['country','co2','state','co2_level', 'coal_co2','cement_co2','flaring_co2','gas_co2','oil_co2'], axis=1)
X.info()

# Using feature selection to select best 5 attributes
selection=SelectKBest(score_func=f_classif, k=5)

# Run feasture seletion
X_selected=selection.fit_transform(X,Y)

# Get the index
feature_selected_idx=selection.get_support(indices=True)

# Print columns
X.columns[feature_selected_idx]
# It seems that primary_energy_consumption has strong relationship to target variable so we need to remove

# For objective2
Y2=selected_data2['co2']
X2=selected_data2.drop(['country','co2','state','co2_level','year', 'methane', 'nitrous_oxide', 'population', 'gdp','primary_energy_consumption'], axis=1)
X2.info()

# Using feature selection to select best 5 attributes
selection2=SelectKBest(score_func=f_classif, k=4)

# Run feasture seletion
X2_selected=selection2.fit_transform(X2,Y2)

# Get the index
feature_selected2_idx=selection2.get_support(indices=True)

# Print columns
X2.columns[feature_selected2_idx]

# Test correlations among selected attributes
correlation_4=round(selected_data2[['co2','year','population','gdp','methane','nitrous_oxide','primary_energy_consumption']].corr(),2)
correlation_5=round(selected_data2[['co2','coal_co2','cement_co2','flaring_co2','gas_co2','oil_co2']].corr(),2)
# It seems that oil_co2 has strong relationship to target variable so we need to remove

# Reduce unnecessary attributes
selected_data2=selected_data2.drop(['country','state','oil_co2','primary_energy_consumption'], axis=1) #vertical reduce variables that unrelated to objective

# Convert categorical to numeric variable
co2_level_map={'Low CO2':0, 'High CO2':1} 
selected_data2['co2_level_int']=selected_data2.co2_level.map(co2_level_map)
selected_data2.info()

# 05-MS
# For objective1, using Regression method

# For objective2, using classification method

# 06-AS
# Test data integrity and quality for objective2
selected_data2[['co2_level_int','coal_co2','cement_co2','flaring_co2','gas_co2']].isnull().sum()

selected_data2[['coal_co2','cement_co2','flaring_co2','gas_co2']].plot.box()

# For objective1, using Random Forest algorithm
from sklearn.ensemble import RandomForestRegressor

# Define dependent and independent variables
Y3=selected_data2['co2']
X3=selected_data2.drop(['co2','co2_level','co2_level_int','coal_co2','cement_co2','flaring_co2','gas_co2'], axis=1)

# For objective2, using Logistic algorithm
import statsmodels.api as sm

# Define dependent and independent variables
Y4=selected_data2['co2_level_int']
X4=selected_data2.drop(['co2','co2_level','co2_level_int','year','population','gdp','methane','nitrous_oxide'], axis=1)

# Add intercept term to the predictors
X4=sm.add_constant(X4)

# 07-DM
# Train the data
from sklearn.model_selection import train_test_split

# For objective1, split data into training and test sets - use 20% for the test set and 80% for the train set; use a fixed seed 200
X3_train,X3_test,Y3_train,Y3_test = train_test_split(X3,Y3,test_size=0.2, random_state=200)

# Test training and test sets
len(X3_train)/len(selected_data2) #train set
len(X3_test)/len(selected_data2) #test set

# Train the RandomForest
random_forest_model=RandomForestRegressor(n_estimators=200, random_state=0)
random_forest_model.fit(X3_train,Y3_train) #train model
random_forest_predictor=random_forest_model.predict(X3_test) #test model
random_forest_error=random_forest_predictor-Y3_test

# Evaluate regresson performance
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(Y3_test, random_forest_predictor))
print('Mean Squared Error:', metrics.mean_squared_error(Y3_test, random_forest_predictor))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y3_test, random_forest_predictor)))

# Draw test plot
import matplotlib.pyplot as plt

plt.figure(1)
plt.clf()
ax=plt.axes(aspect='equal')
plt.scatter(Y3_test,random_forest_predictor)
plt.xlabel('True Values')
plt.ylabel('Predictions')
Lims=[0,140]
plt.xlim(Lims)
plt.ylim(Lims)
plt.plot(Lims,Lims)
plt.grid(False)

plt.figure(2)
plt.clf()
plt.hist(random_forest_error,bins=30)
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.grid(False)

# Verify the accuracy
import scipy.stats as stats

random_forest_pearson_r=stats.pearsonr(Y3_test,random_forest_predictor)
random_forest_R2=metrics.r2_score(Y3_test,random_forest_predictor)
random_forest_RMSE=metrics.mean_squared_error(Y3_test,random_forest_predictor)**0.5
print('Pearson correlation coefficient is {0}, and RMSE is {1}.'.format(random_forest_pearson_r[0],random_forest_RMSE))

# Variable Importance
train_X_column_name=list(X3_train.columns) #find the column name

random_forest_importance=list(random_forest_model.feature_importances_)
random_forest_feature_importance=[(feature,round(importance,8)) 
                                  for feature, importance in zip(train_X_column_name,random_forest_importance)]
random_forest_feature_importance=sorted(random_forest_feature_importance,key=lambda x:x[1],reverse=True)
plt.figure(3)
plt.clf()
importance_plot_x_values=list(range(len(random_forest_importance)))
plt.bar(importance_plot_x_values,random_forest_importance,orientation='vertical')
plt.xticks(importance_plot_x_values,train_X_column_name,rotation='vertical')
plt.xlabel('Variable')
plt.ylabel('Importance')
plt.title('Variable Importances')

# For objective2, split data into training and test sets - use 20% for the test set and 80% for the train set; use a fixed seed 200
X4_train,X4_test,Y4_train,Y4_test = train_test_split(X4,Y4,test_size=0.2, random_state=200)

# Test training and test sets
len(X4_train)/len(selected_data2) #train set
len(X4_test)/len(selected_data2) #test set

# Train the Logistic
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

log_reg=LogisticRegression()
log_train=log_reg.fit(X4_train,Y4_train) #train model
log_predictor=log_reg.predict(X4_test) #test model

# Verify the accuracy
accuracy=accuracy_score(Y4_test,log_predictor)
print('Accuracy:', accuracy)

# Fit logistic regression model
logit_model = sm.Logit(Y4, X4)
result = logit_model.fit()

# Print model summary
print(result.summary())
print("LL", result.llf)
print("AIC:", result.aic)
print("BIC:", result.bic)

# Visulations
import seaborn as sns

sns.boxplot(x=selected_data2['co2_level_int'], y=selected_data2["coal_co2"],linewidth=5) #box plot for coal_co2
plt.show()

sns.boxplot(x=selected_data2['co2_level_int'], y=selected_data2["cement_co2"],linewidth=5) #box plot for cement_co2
plt.show()

sns.boxplot(x=selected_data2['co2_level_int'], y=selected_data2["flaring_co2"],linewidth=5) #box plot for flaring_co2
plt.show()

sns.boxplot(x=selected_data2['co2_level_int'], y=selected_data2["gas_co2"],linewidth=5) #box plot for gas_co2
plt.show()

import matplotlib.pyplot as plt

sns.set_style('white',{'font.sans-serif':['simhei','Arial']})
sns.pairplot(X4) #pair plot

# 08-Iteration
# For objective1
# Define dependent and independent variables
Y5=selected_data2['co2']
X5=X.drop(['year'], axis=1)

# For objective1, split data into training and test sets - use 20% for the test set and 80% for the train set; use a fixed seed 200
X5_train,X5_test,Y5_train,Y5_test = train_test_split(X5,Y5,test_size=0.2, random_state=200)

# Train the OLS
linear_regression = sm.OLS(Y5,sm.add_constant(X5)).fit()
print(linear_regression.summary())

# Calculate fitted values
fitted_values2 = linear_regression.fittedvalues

# Calculate residuals
residuals2 = linear_regression.resid

# Residuals vs. Fitted Values plot
plt.scatter(fitted_values2, residuals2)
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# Train the RandomForest
random_forest_model2=RandomForestRegressor(n_estimators=200, random_state=0)
random_forest_model2.fit(X5_train,Y5_train) #train model
random_forest_predictor2=random_forest_model2.predict(X5_test) #test model
random_forest_error2=random_forest_predictor2-Y5_test

# Evaluate regresson performance
print('Mean Absolute Error:', metrics.mean_absolute_error(Y5_test, random_forest_predictor2))
print('Mean Squared Error:', metrics.mean_squared_error(Y5_test, random_forest_predictor2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y5_test, random_forest_predictor2)))

# Draw test plot
plt.figure(1)
plt.clf()
ax=plt.axes(aspect='equal')
plt.scatter(Y5_test,random_forest_predictor2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
Lims=[0,140]
plt.xlim(Lims)
plt.ylim(Lims)
plt.plot(Lims,Lims)
plt.grid(False)

plt.figure(2)
plt.clf()
plt.hist(random_forest_error2,bins=30)
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.grid(False)

# Verify the accuracy
random_forest_pearson_r2=stats.pearsonr(Y3_test,random_forest_predictor)
random_forest_R3=metrics.r2_score(Y5_test,random_forest_predictor2)
random_forest_RMSE2=metrics.mean_squared_error(Y5_test,random_forest_predictor2)**0.5
print('Pearson correlation coefficient is {0}, and RMSE is {1}.'.format(random_forest_pearson_r[0],random_forest_RMSE2))

# Variable Importance
train_X_column_name2=list(X5_train.columns) #find the column name

random_forest_importance2=list(random_forest_model2.feature_importances_)
random_forest_feature_importance2=[(feature,round(importance,8)) 
                                  for feature, importance in zip(train_X_column_name2,random_forest_importance2)]
random_forest_feature_importance2=sorted(random_forest_feature_importance2,key=lambda x:x[1],reverse=True)
plt.figure(3)
plt.clf()
importance_plot_x_values2=list(range(len(random_forest_importance2)))
plt.bar(importance_plot_x_values2,random_forest_importance2,orientation='vertical')
plt.xticks(importance_plot_x_values2,train_X_column_name2,rotation='vertical')
plt.xlabel('Variable')
plt.ylabel('Importance')
plt.title('Variable Importances')

# For objective2
# Define dependent and independent variables
Y6=selected_data2['co2_level_int']
X6=X2

# For objective2, split data into training and test sets - use 20% for the test set and 80% for the train set; use a fixed seed 200
X6_train,X6_test,Y6_train,Y6_test = train_test_split(X6,Y6,test_size=0.2, random_state=200)

# Train the Logistic
log_reg2=LogisticRegression()
log_train2=log_reg.fit(X6_train,Y6_train) #train model
log_predictor2=log_reg.predict(X6_test) #test model

# Verify the accuracy
accuracy2=accuracy_score(Y6_test,log_predictor2)
print('Accuracy:', accuracy2)

# Fit logistic regression model
logit_model2 = sm.Logit(Y6, X6)
result2 = logit_model2.fit()

# Print model summary
print(result2.summary())
print("LL", result2.llf)
print("AIC:", result2.aic)
print("BIC:", result2.bic)

# Visulations
sns.boxplot(x=Y6, y=X6['oil_co2'],linewidth=5) #box plot for oil_co2
plt.show()

sns.set_style('white',{'font.sans-serif':['simhei','Arial']})
sns.pairplot(X6) #pair plot
