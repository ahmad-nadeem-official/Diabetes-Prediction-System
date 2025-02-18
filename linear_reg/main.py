import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

data = pd.read_csv(r'/home/muhammad-ahmad-nadeem/Projects/supervised-learning-series/linear_reg/resources/diabetes.csv')

# data.head(3)
#    Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age  Outcome
# 0            6      148             72             35        0  33.6                     0.627   50        1
# 1            1       85             66             29        0  26.6                     0.351   31        0
# 2            8      183             64              0        0  23.3                     0.672   32        1

'''
visualization
'''

# plt.figure(figsize=(8, 6))
# sns.boxplot(x=data["Outcome"], y=data["Pregnancies"], palette="Set2")
# plt.title("Pregnancies vs Outcome")
# plt.xlabel("Diabetes Outcome (0 = No, 1 = Yes)")
# plt.ylabel("Number of Pregnancies")
# plt.show()

# plt.figure(figsize=(8,6))
# sns.scatterplot(x=data["Glucose"], y=data["BMI"], hue=data["Outcome"], palette="coolwarm", alpha=0.7)
# plt.title("Glucose vs BMI (Color = Outcome)")
# plt.xlabel("Glucose Level")
# plt.ylabel("BMI")
# plt.show()

# plt.figure(figsize=(10, 8))
# sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
# plt.title("Feature Correlation Heatmap")
# plt.show()

# data.hist(figsize=(12, 8), bins=20, color='skyblue', edgecolor='black')
# plt.suptitle("Feature Distributions", fontsize=16)
# plt.show()

# plt.figure(figsize=(12, 6))
# sns.boxplot(data=data, palette="Set2")
# plt.xticks(rotation=45)
# plt.title("Boxplot of Features")
# plt.show()

# plt.figure(figsize=(8, 6))
# sns.boxplot(x=data["Outcome"], y=data["Pregnancies"], palette="Set2")
# plt.title("Pregnancies vs Outcome")
# plt.xlabel("Diabetes Outcome (0 = No, 1 = Yes)")
# plt.ylabel("Number of Pregnancies")
# plt.show()

# xi = data['Age']
# plt.figure(figsize=(20,5))
# plt.hist(xi,bins=50, color='skyblue', edgecolor='black')
# plt.show()

'''
data cleaning
'''

data.isnull().sum()
# Pregnancies                 0
# Glucose                     0
# BloodPressure               0
# SkinThickness               0
# Insulin                     0
# BMI                         0
# DiabetesPedigreeFunction    0
# Age                         0
# Outcome                     0

# data['Pregnancies']  = data['Pregnancies'].where(~data['Pregnancies'].duplicated()).bfill()
# data['Glucose']  = data['Glucose'].where(~data['Glucose'].duplicated()).bfill
# # data['BloodPressure']  = data['BloodPressure'].where(~data['BloodPressure'].duplicated()).bfill()
# data['SkinThickness']  = data['SkinThickness'].where(~data['SkinThickness'].duplicated()).bfill()
# data['Insulin']  = data['Insulin'].where(~data['Insulin'].duplicated()).bfill()
# # data['BMI']  = data['BMI'].where(~data['BMI'].duplicated()).bfill()
# data['DiabetesPedigreeFunction']  = data['DiabetesPedigreeFunction'].where(~data['DiabetesPedigreeFunction'].duplicated()).bfill()
# data['Age']  = data['Age'].where(~data['Age'].duplicated()).bfill()
# # 0 duplicate value


# data.hist(figsize=(12, 8), bins=20, color='skyblue', edgecolor='black')
# plt.suptitle("Feature Distributions", fontsize=16)
# plt.show()

'''outlier removing
#bloodpressure,  #bmi, #age, #insuline
'''

sns.displot(data=data['Insulin'], bins= 100)
plt.xticks(ticks=range(0, int(data['Insulin'].max()) + 50, 50), rotation=30)
# plt.show()

# q1 = data['BloodPressure'].quantile(0.25)
# q3 = data['BloodPressure'].quantile(0.75)
# iqr = q3 - q1
# max = q3 + (1.5*iqr)
# min = q1 - (1.5*iqr) 
# print(min, max)

# data.describe()
#        Pregnancies  BloodPressure  SkinThickness     Insulin         BMI  DiabetesPedigreeFunction         Age     Outcome
# count   299.000000     692.000000     580.000000  766.000000  748.000000                764.000000  675.000000  768.000000
# mean     13.892977      61.851156      49.456897  209.644909   33.627273                  0.529403   62.216296    0.348958
# std       2.954109      33.466005      31.926944  160.193549   10.241738                  0.357178   11.755826    0.476951
# min       0.000000       0.000000       0.000000    0.000000    0.000000                  0.078000   21.000000    0.000000
# 25%      12.000000      24.000000      17.000000   79.000000   25.375000                  0.259000   61.000000    0.000000
# 50%      14.000000      52.000000      48.000000  183.000000   32.450000                  0.445000   67.000000    0.000000
# 75%      15.000000      98.000000      63.000000  293.000000   41.200000                  0.706000   70.000000    1.000000
# max      17.000000     122.000000      99.000000  846.000000   67.100000                  2.420000   81.000000    1.000000

max_b = 122.000000
data= data[data['BloodPressure']<= max_b]

max_bmi = 67.100000
data= data[data['BMI']<= max_bmi]

max_age = 81.000000
data= data[data['Age'] <= max_age]

max_i = 846.000000
data= data[data['Insulin'] <= max_i]

plt.figure(figsize=(12,10))
sns.boxplot(data=data, palette="Set2")
plt.xticks(rotation = 45)
# plt.show()

scaler = StandardScaler()
# data[['Pregnancies', 'BMI', 'Age', 'Insulin']] = scaler.fit_transform(data[['Pregnancies', 'BMI', 'Age', 'Insulin']])

data.hist(figsize=(12, 8), bins=20, color='skyblue', edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
# plt.show()

# data['Pregnancies'] = data['Pregnancies'].where(~data['Pregnancies'].isnull()).ffill()
# data['SkinThickness'] = data['SkinThickness'].where(~data['SkinThickness'].isnull()).ffill()
data = data.apply(pd.to_numeric, errors='coerce')
data = data.fillna(1)

x = data.drop(columns=['Outcome']).iloc[::-1].reset_index(drop=True)
y = data['Outcome'].iloc[::-1].reset_index(drop=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lr = LinearRegression(fit_intercept=True, n_jobs=-1, copy_X=True, positive=False)
lr.fit(x_train, y_train)
lr.score(x_test, y_test)*100

ndata =np.array([[0,6,148,72,35,0,33.6,0.627]])
ndata_df = pd.DataFrame(ndata, columns=x.columns)
# pr = lr.predict(ndata_df)

joblib.dump(lr, 'trained_model.pkl')
print("model saved")

# print(x_train.dtypes)  # Check data types
# print(x_train.isnull().sum())
