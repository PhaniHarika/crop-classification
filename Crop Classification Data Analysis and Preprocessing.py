#CRISP-[MQ]-Cross Industry Standard Process for Machine Learning Quality
#6 stages:
    #1.Business and data understanding
    #2.Data preparation
    #2.1 exploratory data analysis
    #2.2 data cleaning
    #2.3 feature engineering
    #3. Model Building
    #4.evaluation
    #5.deployment
    #6.monitoring and maintanance
#========================================GROUP 5=============================================#
#=====================================CROP_CLASSIFICATION====================================#
#Exploratory Data Analysis
import pandas as pd #Used for Data Manupulation
#Reads the Dataset
crop_data=pd.read_excel(r"C:/Users/PhaniHarikaSoma24/Downloads/crop.csv.xlsx")

#Display the shape of the Data
crop_data.shape()
#Displays the Top Records
crop_data.head()
#Display the Bottom Records
crop_data.tail()
#Displays the number of rows, index
crop_data.info()
#Displays discriptive statistics and wont work for non-numeric columns
crop_data.describe()



#first moment business decisions(Returns the center value of the data)
#Display the center value
print("Mean") 
crop_data.N.mean()
crop_data.P.mean()
crop_data.K.mean()
crop_data.temperature.mean()
crop_data.humidity.mean()
crop_data.ph.mean()
crop_data.rainfall.mean()

#Displays Middle Most Value
print("Median")
crop_data.N.median()
crop_data.P.median()
crop_data.K.median()
crop_data.temperature.median()
crop_data.humidity.median()
crop_data.ph.median()
crop_data.rainfall.median()

#Displays Most Frequently occured value
print("Mode")
crop_data.N.mode()
crop_data.P.mode()
crop_data.K.mode()
crop_data.temperature.mode()
crop_data.humidity.mode()
crop_data.ph.mode()
crop_data.rainfall.mode()
crop_data.label.mode()



#second moment Business Decision(Spread from the center)
# Variance Doesn't show the actual spread
print("Variance")
crop_data.N.var()
crop_data.P.var()
crop_data.K.var()
crop_data.temperature.var()
crop_data.humidity.var()
crop_data.ph.var()
crop_data.rainfall.var()

#Displays the square root of variance
print("Standard Deviation")
crop_data.N.std()
crop_data.P.std()
crop_data.K.std()
crop_data.temperature.std()
crop_data.humidity.std()
crop_data.ph.std()
crop_data.rainfall.std()

#Difference Between minimum and maximum values
print("Range")
max(crop_data.N)-min(crop_data.N)
max(crop_data.P)-min(crop_data.P)
max(crop_data.K)-min(crop_data.K)
max(crop_data.temperature)-min(crop_data.temperature)
max(crop_data.humidity)-min(crop_data.humidity)
max(crop_data.ph)-min(crop_data.ph)
max(crop_data.rainfall)-min(crop_data.rainfall)



#Third moment Business Decision(Direction of spread of business)
# Skewness is of 3 types
# 1. Positive Skew(Right Skew)-When long tail is at right side
# 2. Negative Skew(Left Skew)-when long tail is at left side
# 3. N0 skew- when data is normally distributed
print("skewness")
crop_data.N.skew()
crop_data.P.skew()
crop_data.K.skew()
crop_data.temperature.skew()
crop_data.humidity.skew()
crop_data.ph.skew()
crop_data.rainfall.skew()



#Fourth moment Business Decision(Captures the peakedness of the Distribution)
# Kurtosis is of 3 types
# 1. Positive Kurtosis- If the peak is sharp & data is concentrated at single point
# 2. Negative Kurtosis-If peak is wide and data is not concentrated at single pont
# 3. It Represents Normal Distribution
print("Kurtosis")
crop_data.N.kurt()
crop_data.P.kurt()
crop_data.K.kurt()
crop_data.temperature.kurt()
crop_data.humidity.kurt()
crop_data.ph.kurt()
crop_data.rainfall.kurt()


#========Graphical Representation==========
#importing necessary Packages
import pandas as pd
import matplotlib.pyplot as plt # It is the basic package used for data visualization
import seaborn as sns #It is an advanced package,Built on top of matplotlib

#Reading the dataset
crop_data=pd.read_excel(r"C:/Users/PhaniHarikaSoma24/Downloads/crop.csv.xlsx")
   
#Univariate plots
# 1.Univarieate-visiualizing single column at a time
# 2.Bivarieate-Two columns at a time

#Histogram for data distribution #we visualization understand data distribution
plt.hist(crop_data.N)
plt.hist(crop_data.P)
plt.hist(crop_data.K)
plt.hist(crop_data.temperature)
plt.hist(crop_data.humidity)
plt.hist(crop_data.ph)
plt.hist(crop_data.rainfall)

#Boxplot is used to Display outlayers
# Lower limit[Q1]=Q1-1.5(IQR)
# Higher limit[Q2]=Q3+1.5(IQR)
# IQR=Q3-Q1
plt.boxplot(crop_data.temperature)
plt.boxplot(crop_data.humidity)
plt.boxplot(crop_data.ph)
plt.boxplot(crop_data.rainfall)

#Scatter plot is used to understand the relation between 2 variables
# The strength of the scatter plot lies on the distance between the plots
# If r>0.85 Strength is strong
# If r<0.4 Strength is weak
# If r lies between 0.4 to 0.85 strength is moderate
plt.figure(figsize=(10,8))#set figure size
plt.scatter(x=crop_data.temperature,y=crop_data.rainfall,color="Blue")
plt.xlabel("temperature") #x-Axis
plt.ylabel("rainfall") #y-Axis
plt.show()

plt.figure(figsize=(10,8))#set figure size
plt.scatter(x=crop_data.temperature,y=crop_data.humidity,color="Blue")
plt.xlabel("temperature") #x-Axis
plt.ylabel("humidity") #y-Axis
plt.show()

plt.figure(figsize=(10,8))#set figure size
plt.scatter(x=crop_data.temperature,y=crop_data.ph,color="Blue")
plt.xlabel("temperature") #x-Axis
plt.ylabel("ph") #y-Axis
plt.show()

plt.figure(figsize=(10,8))#set figure size
plt.scatter(x=crop_data.N,y=crop_data.P,color="Blue")
plt.xlabel("N") #x-Axis
plt.ylabel("P") #y-Axis
plt.show()

plt.figure(figsize=(10,8))#set figure size
plt.scatter(x=crop_data.N,y=crop_data.K,color="Blue")
plt.xlabel("N") #x-Axis
plt.ylabel("K") #y-Axis
plt.show()

plt.figure(figsize=(10,8))#set figure size
plt.scatter(x=crop_data.N,y=crop_data.temperature,color="Blue")
plt.xlabel("N") #x-Axis
plt.ylabel("temperature") #y-Axis
plt.show()

plt.figure(figsize=(10,8))#set figure size
plt.scatter(x=crop_data.N,y=crop_data.humidity,color="Blue")
plt.xlabel("N") #x-Axis
plt.ylabel("humidity") #y-Axis
plt.show()

plt.figure(figsize=(10,8))#set figure size
plt.scatter(x=crop_data.N,y=crop_data.ph,color="Blue")
plt.xlabel("N") #x-Axis
plt.ylabel("ph") #y-Axis
plt.show()

plt.figure(figsize=(10,8))#set figure size
plt.scatter(x=crop_data.N,y=crop_data.rainfall,color="Blue")
plt.xlabel("N") #x-Axis
plt.ylabel("rainfall") #y-Axis
plt.show()

plt.figure(figsize=(10,8))#set figure size
plt.scatter(x=crop_data.P,y=crop_data.K,color="Blue")
plt.xlabel("P") #x-Axis
plt.ylabel("K") #y-Axis
plt.show()

plt.figure(figsize=(10,8))#set figure size
plt.scatter(x=crop_data.P,y=crop_data.temperature,color="Blue")
plt.xlabel("P") #x-Axis
plt.ylabel("temperature") #y-Axis
plt.show()

plt.figure(figsize=(10,8))#set figure size
plt.scatter(x=crop_data.P,y=crop_data.humidity,color="Blue")
plt.xlabel("P") #x-Axis
plt.ylabel("humidity") #y-Axis
plt.show()

plt.figure(figsize=(10,8))#set figure size
plt.scatter(x=crop_data.P,y=crop_data.ph,color="Blue")
plt.xlabel("P") #x-Axis
plt.ylabel("ph") #y-Axis
plt.show()

plt.figure(figsize=(10,8))#set figure size
plt.scatter(x=crop_data.P,y=crop_data.rainfall,color="Blue")
plt.xlabel("P") #x-Axis
plt.ylabel("rainfall") #y-Axis
plt.show()

plt.figure(figsize=(10,8))#set figure size
plt.scatter(x=crop_data.K,y=crop_data.temperature,color="Blue")
plt.xlabel("K") #x-Axis
plt.ylabel("temperature") #y-Axis
plt.show()
plt.figure(figsize=(10,8))#set figure size
plt.scatter(x=crop_data.K,y=crop_data.humidity,color="Blue")
plt.xlabel("K") #x-Axis
plt.ylabel("humidity") #y-Axis
plt.show()

plt.figure(figsize=(10,8))#set figure size
plt.scatter(x=crop_data.K,y=crop_data.ph,color="Blue")
plt.xlabel("K") #x-Axis
plt.ylabel("ph") #y-Axis
plt.show()

plt.figure(figsize=(10,8))#set figure size
plt.scatter(x=crop_data.K,y=crop_data.rainfall,color="Blue")
plt.xlabel("K") #x-Axis
plt.ylabel("rainfall") #y-Axis
plt.show()

plt.figure(figsize=(10,8))#set figure size
plt.scatter(x=crop_data.K,y=crop_data.ph,color="Blue")
plt.xlabel("humidity") #x-Axis
plt.ylabel("ph") #y-Axis
plt.show()

plt.figure(figsize=(10,8))#set figure size
plt.scatter(x=crop_data.K,y=crop_data.rainfall,color="Blue")
plt.xlabel("humidity") #x-Axis
plt.ylabel("rainfall") #y-Axis
plt.show()

plt.figure(figsize=(10,8))#set figure size
plt.scatter(x=crop_data.ph,y=crop_data.rainfall,color="Blue")
plt.xlabel("ph") #x-Axis
plt.ylabel("rainfall") #y-Axis
plt.show()
#data cleaning-5-sub-phases
   #1.Handling Duplicates
   #2.OuterLayer Treatment/Treating the outlayers
   #3.Dummy variable craetion/encoding
   #4.Handling missing values
   #5.Feature scaling/Feature shrinking
#Importing necessary Packages

import pandas as pd
#Finding duplicates rows in the DataFrame
duplicate=crop_data.duplicated()
#Identifies Whether they are duplicates or not
duplicate
#counting the total number of duplicate row
total_duplicates=sum(duplicate)
total_duplicates
#Removing duplicates rows from the dataframe
data_first=crop_data.drop_duplicates()
data_first

#outlayer Treatment
#outlayer treatmet can significantly affect model performance
#Handling outlayer ensures
#Importng necessary libraries

import pandas as pd
#To calculate complex mathematical problems
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df= pd.read_excel(r"C:/Users/PhaniHarikaSoma24/Downloads/crop.csv.xlsx")
plt.boxplot(df.N)
plt.boxplot(df.P)
plt.boxplot(df.K)
plt.boxplot(df.temperature)
plt.boxplot(df.humidity)
plt.boxplot(df.ph)
plt.boxplot(df.rainfall)


#winsorization
#pip install feature-engine
from feature_engine.outliers import Winsorizer
#IQR method for Winsorization
winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['temperature'])
df['temperature_IQR'] = winsor.fit_transform(df[['temperature']])
sns.boxplot(df['temperature_IQR'])
plt.show()
winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['humidity'])
df['humidity_IQR'] = winsor.fit_transform(df[['humidity']])
sns.boxplot(df['humidity_IQR'])
plt.show()
winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['ph'])
df['ph'] = winsor.fit_transform(df[['ph']])
sns.boxplot(df['ph'])
plt.show()
winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['rainfall'])
df['rainfall_IQR'] = winsor.fit_transform(df[['rainfall']])
sns.boxplot(df['rainfall_IQR'])
plt.show()

#creating the Boxplot to visualize the distribution
#pypi-centralized repository
#If feature_engine is not found go to the console and type pip install feature_engine
#DUMY VARIABLES-converting non-numeric to numeric
#importing necessary packages

import pandas as pd
df= pd.read_excel(r"C:/Users/PhaniHarikaSoma24/Downloads/crop.csv.xlsx")
#dropping make ,model
df.drop(['label'],axis=1,inplace=True)
#Creating dummy variables for categorial columns
df_new=pd.get_dummies(df)
df_new_1=pd.get_dummies(df,drop_first=True)


#HANDLING MISSING VALUES
#importing necessary libraries
import pandas as pd
import numpy as np
df= pd.read_excel(r"C:/Users/PhaniHarikaSoma24/Downloads/crop.csv.xlsx")
#Displaying the dataset information
df.info()
#Checking for the count of missing values
print(df.isna().sum())
from sklearn.impute import SimpleImputer
#Mean Imputation

mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
df["N"]=mean_imputer.fit_transform(df[["N"]])
df["N"].isna().sum()
mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
df["P"]=mean_imputer.fit_transform(df[["P"]])
df["P"].isna().sum()
mean_imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
df["K"]=mean_imputer.fit_transform(df[["K"]])
df["K"].isna().sum()

#Median Imputation

median_imputer=SimpleImputer(missing_values=np.nan,strategy='median')
df["N"]=mean_imputer.fit_transform(df[["N"]])
df["N"].isna().sum()
median_imputer=SimpleImputer(missing_values=np.nan,strategy='median')
df["P"]=mean_imputer.fit_transform(df[["P"]])
df["P"].isna().sum()
median_imputer=SimpleImputer(missing_values=np.nan,strategy='median')
df["K"]=mean_imputer.fit_transform(df[["K"]])
df["K"].isna().sum()

#Mode Imputation

mode_imputer=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
df["N"]=mode_imputer.fit_transform(df[["N"]]).ravel()
df["N"].isna().sum()
df["P"]=mode_imputer.fit_transform(df[["P"]]).ravel()
df["P"].isna().sum()
df["K"]=mode_imputer.fit_transform(df[["K"]]).ravel()
df["K"].isna().sum()

#feature scaling(or) Feature Shrinking
#import necessary packages

import pandas as pd
data=pd.read_excel(r"C:/Users/PhaniHarikaSoma24/Downloads/crop.csv.xlsx")   
#Generating descriptive statistics  
des=data.describe()
data1=data.loc[:,['N','P','K','temperature','humidity','ph','rainfall']]
#Standardization
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
#scaling the data using standardscaler
df_crop=scaler.fit_transform(data1)
#converting the scaled array back to a DataFrame
dataset_crop=pd.DataFrame(df_crop)
#Assigning the column names to the dataframe
# converting the scaled array back to a DataFrame
dataset_crop = pd.DataFrame(df_crop, columns=['N','P','K','temperature','humidity','ph','rainfall'])

# descriptive stats
res_crop = data.describe()  # note the parentheses

# concatenate original and scaled data
scaled_data_crop = pd.concat([data.loc[:, ['N','P','K','temperature','humidity','ph','rainfall']], dataset_crop], axis=1)

res_crop=scaled_data_crop.describe()

#normalization
from sklearn.preprocessing import MinMaxScaler
#initialise the minmaxscaler
minmaxScale=MinMaxScaler()
df_norm=minmaxScale.fit_transform(data1)
#converting the scaled array
dataset_norm=pd.DataFrame(df_norm)
dataset_norm.columns=['N','P','K','temperature','humidity','ph','rainfall']
res_std=dataset_norm.describe
scaled_data_norm=pd.concat([data.loc[:,'N','P','K','temperature','humidity','ph','rainfall'],dataset_norm],axis=1)
res_norm=scaled_data_norm.describe()






