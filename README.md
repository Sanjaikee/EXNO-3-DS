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
# NAME : AARON RAJESH . R
# REG.NO: 212223100001
         import pandas as pd
     df=pd.read_csv("/content/Encoding Data.csv")
     df
    ![image](https://github.com/Aaron-0111/EXNO-3-DS/assets/149347631/fdfd5ee1-8f8a-4b1e-a866-6aa806dd0700)
     from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
    ![image](https://github.com/Aaron-0111/EXNO-3-DS/assets/149347631/7641797d-1ceb-4bd3-8876-7a6d43233960)
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
![image](https://github.com/Aaron-0111/EXNO-3-DS/assets/149347631/68b28e89-92c6-4666-9079-bd1256427e7c)
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
![image](https://github.com/Aaron-0111/EXNO-3-DS/assets/149347631/9f8004d1-7222-4b6a-aaca-f324cf136d9a)
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
![image](https://github.com/Aaron-0111/EXNO-3-DS/assets/149347631/1f350be5-2ea2-496b-a241-853ace1edba8)
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
![image](https://github.com/Aaron-0111/EXNO-3-DS/assets/149347631/43dd6166-0bf5-4457-9703-7891ca76aaef)
pd.get_dummies(df2,columns=["nom_0"])
![image](https://github.com/Aaron-0111/EXNO-3-DS/assets/149347631/67264de2-19a2-4edf-9bf4-5ff39e05d96e)
pip install --upgrade category_encoders
![image](https://github.com/Aaron-0111/EXNO-3-DS/assets/149347631/4302c655-b42e-4c15-8e37-70a4ff181cc2)
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
fb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
![image](https://github.com/Aaron-0111/EXNO-3-DS/assets/149347631/63a036e9-fae9-4f47-b64a-2107223941c1)
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
![image](https://github.com/Aaron-0111/EXNO-3-DS/assets/149347631/c4349627-63e8-4395-ad95-af8c7593479b)
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
![image](https://github.com/Aaron-0111/EXNO-3-DS/assets/149347631/a560cbf3-c861-49c8-9435-72f0a35cf186)
df.skew()
![image](https://github.com/Aaron-0111/EXNO-3-DS/assets/149347631/c46a816b-a1a8-4115-9b48-23ff28d455e8)
np.log(df["Highly Positive Skew"])
![image](https://github.com/Aaron-0111/EXNO-3-DS/assets/149347631/b4548b87-b6c5-43bf-9fce-fab957681365)
np.reciprocal(df["Moderate Positive Skew"])
![image](https://github.com/Aaron-0111/EXNO-3-DS/assets/149347631/bec244b1-6f9c-49e1-a707-d9d200ee3eae)
np.sqrt(df["Highly Positive Skew"])
![image](https://github.com/Aaron-0111/EXNO-3-DS/assets/149347631/1e8f60bc-dca2-4d40-b60a-2022b703bf6e)
np.square(df["Highly Positive Skew"])
![image](https://github.com/Aaron-0111/EXNO-3-DS/assets/149347631/449d4488-e10b-4ffa-a780-a33ee613666b)
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"]) df
![image](https://github.com/Aaron-0111/EXNO-3-DS/assets/149347631/0b9e38ed-72f0-4779-9826-46e58307da6e)
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
![image](https://github.com/Aaron-0111/EXNO-3-DS/assets/149347631/ee80217f-4f3b-4b6d-940e-2d940c4340a8)
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
![image](https://github.com/Aaron-0111/EXNO-3-DS/assets/149347631/b671092d-9971-4ede-beb1-82f068a4caed)
import matplotlib.pyplot as plt import seaborn as sns import statsmodels.api as sm import scipy.stats as stats

sm.qqplot(df["Moderate Negative Skew"],line='45')

plt.show()
![image](https://github.com/Aaron-0111/EXNO-3-DS/assets/149347631/41896cef-34b9-47e2-832f-084a972224b4)
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
![image](https://github.com/Aaron-0111/EXNO-3-DS/assets/149347631/58ea4a79-8619-440f-bf6e-5eca3e1d3fec)
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
![image](https://github.com/Aaron-0111/EXNO-3-DS/assets/149347631/4fb3b1b1-81c6-437c-afe1-b393f60f8e32)

# RESULT:
        Hence performing Feature Encoding and Transformation process is Successful.

       
