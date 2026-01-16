# Data-visualisation-and-data-analysis-
This folder contains various projects about analysing a data set given and creating a bivariate graph for attributes in the data set using python and machine learning
As u can check the file the code is written in that file and explained detailed 

the following python code is used for analysis of data set 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from google.colab import files
files.upload()

df = pd.read_csv('/content/software_defects.csv')
df.head()
df.info()
df.describe()

The following codegives the bivariate graph 
plt.hist(df['defect_density_per_kloc'], bins=20)
plt.xlabel("Defect Density per KLOC")
plt.ylabel("Frequency")
plt.title("Distribution of Defect Density")
plt.show()


df.isnull().sum() /* Checks for missing values */
# Fill numerical columns with mean
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Fill categorical columns with mode
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)/* handles missing values*/



le = LabelEncoder() /* encode categorical variables*/
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

plt.figure(figsize=(12,8))/* corelation heatmap*/
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

/*complexity vs Defect density*/
plt.scatter(df['code_complexity_cyclomatic'], df['defect_density_per_kloc'])
plt.xlabel("Cyclomatic Complexity")
plt.ylabel("Defect Density per KLOC")
plt.title("Code Complexity vs Defect Density")
plt.show()

/*Development Time vs Defect Density*/
plt.scatter(df['development_time_months'], df['defect_density_per_kloc'])
plt.xlabel("Development Time (Months)")
plt.ylabel("Defect Density per KLOC")
plt.title("Development Time vs Defect Density")
plt.show()

/*Feature Selection*/
X = df.drop('defect_density_per_kloc', axis=1)
y = df['defect_density_per_kloc'

/*Train-Test Splitt*/
X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42

)

/*Linear Regression*/
model = LinearRegression()
model.fit(X_train, y_train)

/*Make predictions*/
y_pred = model.predict(X_test)

/*Model Evaluation*/
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

Expected output for model Evaluation 
Mean Absolute Error: 2.636130847321888
Mean Squared Error: 9.099127879200559
R2 Score: -0.004125892628209504