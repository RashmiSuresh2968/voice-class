# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
#preprocess.
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer
warnings.filterwarnings('ignore')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_train= pd.read_csv('C:/Users/Rashmi S/Downloads/voice.csv (1).zip')
df_train.head()
df_train.shape
df_train.columns
df_train.isnull().sum()
df_train.describe()

def check_outliers(col):
    q1,q3=df[col].quantile([0.25,0.75])
    iqr=q3-q1
    rang=1.5*iqr
    return(q1-rang,q3+rang)
def plot(col):
    fig,axes=plt.subplots(1,2)
    sns.boxplot(data=df,x=col,ax=axes[0])
    sns.distplot(a=df[col],ax=axes[1],color='#ff4125')
    fig.set_size_inches(15,5)
    lower,upper = check_outliers(col)
    l=[df[col] for i in df[col] if i>lower and i<upper] 
    print("Number of data points remaining if outliers removed : ",len(l))
    
df=df_train
del df_train
df.columns
plot('meanfreq')
plot('sd')

plot('median')
plot('Q25')
plot('Q75')
plot('skew')
plot('kurt')
plot('sp.ent')
plot('sfm')
plot('meanfun')
sns.countplot(data=df,x='label')
df['label']=df['label'].replace({'male':1,'female':0})
df.head()

cor_mat= df[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)

def plot_against_target(feature):
    sns.factorplot(data=df,y=feature,x='label',kind='box')
    fig=plt.gcf()
    fig.set_size_inches(7,7)
plot_against_target('meanfreq')
plot_against_target('sd')
plot_against_target('median')
plot_against_target('Q25')
plot_against_target('IQR')
plot_against_target('sp.ent')
plot_against_target('meanfun')
g = sns.PairGrid(df[['meanfreq','sd','median','Q25','IQR','sp.ent','sfm','meanfun','label']], hue = "label")
g = g.map(plt.scatter).add_legend()
for col in df.columns:
    l,u=check_outliers(col)
    df=df[(df[col]>l)&(df[col]<u)]
df.shape
temp_df=df.copy()
temp_df.drop(['skew','kurt','mindom','maxdom','centroid'],axis=1,inplace=True)
temp_df.head()
## skewness with pearson coefficient
temp_df['pear_skew']=temp_df['meanfreq']-temp_df['mode']
temp_df['pear_skew']=temp_df['pear_skew']/temp_df['sd']
temp_df.head(10)

sns.boxplot(data=temp_df,y='pear_skew',x='label');
temp_df['meanfreq']=temp_df['meanfreq'].apply(lambda x:x*2)
temp_df['median']=temp_df['meanfreq']+temp_df['mode']
temp_df['median']=temp_df['median'].apply(lambda x:x/3)
sns.boxplot(data=temp_df,y='median',x='label');
scaler=StandardScaler()
scaled_df=scaler.fit_transform(temp_df.drop('label',axis=1))
X=scaled_df
Y=df['label'].values
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=42)
clf_lr=LogisticRegression()
clf_lr.fit(x_train,y_train)
pred=clf_lr.predict(x_test)
print(accuracy_score(pred,y_test))


