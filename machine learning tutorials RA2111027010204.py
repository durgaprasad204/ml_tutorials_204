#!/usr/bin/env python
# coding: utf-8

# In[35]:


# k_means clustering

import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler # this is to scale our data

from sklearn.cluster import KMeans


# In[5]:


# Step 2 load data
data = pd.read_csv("C:\\Users\\kanak\\Downloads\\hoodmaps.csv")


# In[6]:


data.head()


# In[7]:


data.info()


# In[8]:


data.dtypes


# In[9]:


#Step 3. Transform data in Python
data['Average income'] = data['Average income'].replace('£', '', regex = True)
data['Average income'] = data['Average income'].replace(',', '', regex = True)
data.head()


# In[10]:


data.info()


# In[11]:


data['Average income'] = data['Average income'].astype(float)
data.info()


# In[12]:


data.head()


# In[14]:


#Step 4. Visualise the Data
# create a scatterplot of Age vs Income 

sns.scatterplot(x = data['Average Age'], y = data['Average income'])


# In[16]:


#Step 5. Modelling with K-Means
# to do the k-means we only want the Age and Income information 

data_new = data.drop(['Wards in Hackney'], axis = 1)
data_new.head()


# In[17]:


# to normalise, we use the MinMaxScaler function from sklearn 

data_scaled = MinMaxScaler().fit_transform(data_new)
data_scaled


# In[18]:


# create the KMeans model object with a number of clusters K. 

model = KMeans(n_clusters = 3, random_state = 123)
# fit the model to our scaled data

model.fit(data_scaled)


# In[19]:


model.labels_


# In[20]:


# add a column to the dataframe called 'cluster' which tells us which cluster each data point belongs to 

data['Cluster'] = model.labels_
data.head()


# In[21]:


data.info()


# In[23]:


# Pull wards in hackney where cluster column is equal 0

data[['Wards in Hackney']][data['Cluster'] == 0]


# In[24]:


# Pull wards in hackney where cluster column is equal 1

data[['Wards in Hackney']][data['Cluster'] == 1]


# In[25]:


# Pull wards in hackney where cluster column is equal 2

data[['Wards in Hackney']][data['Cluster'] == 2]


# In[26]:


data[data['Cluster'] == 0]


# In[27]:


#Step 6. Visualise the Clusters
# we want to recreate the scatter plot, but with the data points coloured according to the cluster 

# specify the 'hue' parameter 

sns.scatterplot(x = data['Average Age'], y = data['Average income'], hue = data['Cluster'])


# In[28]:


fig, ax = plt.subplots() # creates a canvas / empty plot to fill 
ax.scatter(x = data['Average Age'], y = data['Average income'])

for i, txt in enumerate(data['Wards in Hackney']):
    ax.annotate(txt, (data['Average Age'][i], data['Average income'][i]))

# enumerate supplies a number for each ward 
# the plot uses these numbers to label the points 


# In[29]:


data.groupby('Cluster')['Average Age'].mean().plot(kind = 'bar')


# In[30]:


data.groupby('Cluster')['Average income'].mean().plot(kind = 'bar')


# In[34]:


sns.lineplot(x=num_clusters, y=score)
plt.xlabel('Number of Clusters k')
plt.ylabel('Score')
plt.title('Line Plot of Score vs. Number of Clusters')
plt.show()


# In[36]:


# k_means clustering ending


# In[37]:


# logistic_regression starting here


# In[38]:


#Import Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import time
import random

random.seed(100)


# In[40]:


wine = pd.read_csv(r"C:\Users\kanak\Downloads\winequality-red.csv")
wine.head()


# In[41]:


from sklearn.preprocessing import LabelEncoder
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)
label_quality = LabelEncoder()
wine['quality'] = label_quality.fit_transform(wine['quality'])
wine['quality'].value_counts()


# In[42]:


sns.countplot(wine['quality'])


# In[43]:


wine.columns


# In[44]:


sns.pairplot(wine)


# In[45]:


wine[wine.columns[:11]].describe()


# In[46]:


wine.isna().any()


# In[47]:


wine.corrwith(wine.quality).plot.bar(
        figsize = (20, 10), title = "Correlation with quality", fontsize = 15,
        rot = 45, grid = True)


# In[48]:


sns.set(style="white")

# Compute the correlation matrix
corr = wine.corr()
corr.head()


# In[49]:


X = wine.drop('quality',axis=1)
y=wine['quality']


# In[50]:


X.head()


# In[51]:


features_label = wine.columns[:11]
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
classifier.fit(X, y)
importances = classifier.feature_importances_
indices = np. argsort(importances)[::-1]
for i in range(X.shape[1]):
    print ("%2d) %-*s %f" % (i + 1, 30, features_label[i],importances[indices[i]]))


# In[52]:


plt.title('Feature Importances')
plt.bar(range(X.shape[1]),importances[indices], color="green", align="center")
plt.xticks(range(X.shape[1]),features_label, rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[53]:


from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 5)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train2 = pd.DataFrame(sc.fit_transform(X_train))
X_test2 = pd.DataFrame(sc.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2
#Using Principal Dimensional Reduction
from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
print(pd.DataFrame(explained_variance))


# In[54]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

results = pd.DataFrame([['Logistic Regression', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
print(results)


# In[55]:


#logistic_regression end here


# In[56]:


# multiple linear regression start here#


# In[73]:


# Multiple Linear Regression

import numpy as np
import pandas as pd

# Importing the datasets
datasets = pd.read_csv(r"C:\Users\kanak\Downloads\50_Startups.csv")
X = datasets.iloc[:, :-1].values
Y = datasets.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categories='auto', drop='first')
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Fitting the Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_Train, Y_Train)

# Predicting the Test set results
Y_Pred = regressor.predict(X_Test)

# Applying backward elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

# Select significance level (e.g., 0.05)
SL = 0.05

X_Optimal = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_Optimal).fit()
while regressor_OLS.pvalues.max() > SL:
    X_Optimal = np.delete(X_Optimal, regressor_OLS.pvalues.argmax(), axis=1)
    regressor_OLS = sm.OLS(endog = Y, exog = X_Optimal).fit()

# Fitting the Multiple Linear Regression to the Optimal Training set
X_Optimal_Train, X_Optimal_Test = train_test_split(X_Optimal, test_size = 0.2, random_state = 0)
regressor.fit(X_Optimal_Train, Y_Train)

# Predicting the Optimal Test set results
Y_Optimal_Pred = regressor.predict(X_Optimal_Test)


# In[74]:


print(Y_Optimal_Pred)


# In[77]:


# Multiple Linear Regression end here


# In[87]:


#random_forest start here


# In[88]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
digits = load_digits()
digits


# In[89]:


dir(digits)


# In[90]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])


# In[93]:


df = pd.DataFrame(data=digits.data,columns=digits.feature_names)
df


# In[94]:


df['target']=digits.target


# In[95]:


df


# In[96]:


x_train,x_test,y_train,y_test = train_test_split(df.drop('target',axis=1),df.target,test_size=0.2)


# In[97]:


model= RandomForestClassifier()


# In[98]:


model.fit(x_train,y_train)


# In[99]:


model.predict(x_test)


# In[100]:


y_test


# In[101]:


model.score(x_test,y_test)


# In[102]:


from sklearn.metrics import confusion_matrix


# In[103]:


cm = confusion_matrix(y_test,model.predict(x_test))
cm


# In[104]:


import seaborn as sns


# In[105]:


sns.heatmap(cm,annot=True,cmap='Greens', fmt='g')
plt.xlabel("Actual")
plt.ylabel("Predicted")


# In[106]:


#random forest ends here


# In[107]:


#svm code starts here


# In[108]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.svm import SVC


# In[109]:


data = load_digits()


# In[110]:


digits


# In[111]:


dir(data)


# In[112]:


df = pd.DataFrame(data = data.data , columns = data.feature_names)


# In[113]:


df


# In[114]:


df['target'] = data['target']


# In[115]:


df


# In[116]:


data.target_names


# In[117]:


df.shape


# In[118]:


df[df['target'] == 1].head()


# In[119]:


X = df.drop('target',axis=1)
X


# In[120]:


y = df.target


# In[121]:


y


# In[122]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[123]:


import matplotlib.pyplot as plt


# In[124]:


df0 = df[df.target == 0]
df1 = df[df.target == 1]


# In[125]:


df0


# In[126]:


df1


# In[127]:


plt.xlabel("pixel_0_2")
plt.ylabel("pixel_7_2")
plt.scatter(df0["pixel_0_2"],df0["pixel_7_2"],color ="g",marker ='+')
plt.scatter(df1["pixel_0_2"],df1["pixel_7_2"],color = "r",marker = ".")
plt.show()


# In[129]:


model = SVC(kernel="linear")


# In[130]:


model.fit(X_train,y_train)


# In[131]:


model.predict(X_test)


# In[132]:


y_test


# In[133]:


model.score(X_test,y_test)


# In[ ]:


#svm#code ends here#

