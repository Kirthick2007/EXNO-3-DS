## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
NAME : KIRTHICK SHA R
DEPARTMENT : 212224230124
```

import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
![image](https://github.com/user-attachments/assets/c9ff95e2-e1a8-47c0-93c2-5cb74e8418dd)

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])

![image](https://github.com/user-attachments/assets/08d6f422-eab3-4e6f-a3a0-040a994c52fe)

df['bo2']=e1.fit_transform(df[["ord_2"]])
df

![image](https://github.com/user-attachments/assets/4bdf3a35-8ea7-4f32-a69f-b95ee65049a6)

le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc

![image](https://github.com/user-attachments/assets/5ebb4575-b43e-48f5-8aa1-a3fd9e1ce82b)

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2

![image](https://github.com/user-attachments/assets/b783fdc8-e75a-4e8c-b182-0cfc7a89325b)

pd.get_dummies(df2,columns=["nom_0"])

![image](https://github.com/user-attachments/assets/10e6b17b-f418-4ac7-aaa6-60697d3784f6)

pip install --upgrade category_encoders
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
dfb=pd.concat([df,nd],axis=1)
dfb
![image](https://github.com/user-attachments/assets/a78f7c15-fbe2-4847-b99d-a6244124012a)

from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC

![image](https://github.com/user-attachments/assets/b13f24ef-ad47-440f-95da-ecd77d666a84)

import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df

![image](https://github.com/user-attachments/assets/6ef9883a-be0e-43b4-90f7-e668beb295f2)

df.skew()
![image](https://github.com/user-attachments/assets/3ed94fb3-ed15-46b9-9e3d-eadfa1c55ded)

np.log(df["Highly Positive Skew"])
![image](https://github.com/user-attachments/assets/921a1804-48e7-4f24-ab13-fbf1d33b46ed)

np.reciprocal(df["Moderate Positive Skew"])
![image](https://github.com/user-attachments/assets/8257b02d-fd0e-4ccf-a009-84ccb48ae96d)

np.sqrt(df["Highly Positive Skew"])
![image](https://github.com/user-attachments/assets/3f8e082e-6518-4647-9e25-ddaf55e737bd)

np.square(df["Highly Positive Skew"])
![image](https://github.com/user-attachments/assets/e3f58e66-698d-47ef-b445-5c877e341f30)

df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
![image](https://github.com/user-attachments/assets/8936fe0d-d180-4343-9ac9-7a6a974ef3ff)

df.skew()
![image](https://github.com/user-attachments/assets/cbdcdf2f-c412-409f-8484-15b80ec9c4f3)

df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"
df.skew()
![image](https://github.com/user-attachments/assets/f1cf7de9-d707-43c6-849a-52ec19778034)

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
![image](https://github.com/user-attachments/assets/3ab79264-514c-41b5-a2e5-0b7a7bfdbea6)

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
![image](https://github.com/user-attachments/assets/47bd6813-f170-4733-8727-410920dadbb1)

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
![image](https://github.com/user-attachments/assets/2c893500-378f-4b29-9013-953bfc0308f4)

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
![image](https://github.com/user-attachments/assets/79e88a90-1173-478e-8143-7f8a871c7ed7)

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
![image](https://github.com/user-attachments/assets/1abff599-616b-45eb-b21a-4df144d07978)

dt=pd.read_csv("titanic_dataset.csv")
dt
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
![image](https://github.com/user-attachments/assets/30dfa736-0fb6-499f-b07f-ed9617416820)

sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
![image](https://github.com/user-attachments/assets/7eed6555-c6e9-45d5-8817-341944b83b33)




# RESULT:
 Thus the given data, Feature Encoding, Transformation process and save the data to a file
was performed successfully.

       
