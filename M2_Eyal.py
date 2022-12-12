#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlrd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression


# In[47]:


data = pd.read_csv('C:/Users/keren/Downloads/bodyfat.csv')
data


# In[48]:


x = data.loc[:, data.columns != "BodyFat"].values
y = data.iloc[:, 1:2].values
alfa_A=[0,0.1,0.25,0.5,1,5]
p = 14


# In[49]:


data_ridge=pd.DataFrame(columns=['alpha','Coefs','RMSE','Adjusted R^2'])
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.30)

for a in alfa_A:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train,y_train)
    pred_ridge = ridge.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test,pred_ridge))
    r2_r=r2_score(y_test, pred_ridge)
    n = len(y_train)
    Adj_r2 = 1-(1-r2_r)*(n-1)/(n-p-1)
    data_ridge = data_ridge.append({'alpha': a,'Coefs': ridge.coef_[0],'RMSE' : rmse, 'Adjusted R^2' : Adj_r2}, ignore_index = True)


# In[50]:


data_ridge


# In[51]:


data_lasso=pd.DataFrame(columns=['alpha','Coefs','RMSE','Adjusted R^2'])

for a in alfa_A:
    lasso_model=Lasso(alpha=a)
    lasso_model.fit(X_train,y_train)
    pred_laso = lasso_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test,pred_laso))
    r2_l=r2_score(y_test, pred_laso)
    n = len(y_train)
    Adj_r2 = 1-(1-r2_l)*(n-1)/(n-p-1)
    data_lasso = data_lasso.append({'alpha': a,'Coefs': lasso_model.coef_,'RMSE' : rmse, 'Adjusted R^2' : r2_l}, ignore_index = True)


# In[52]:


data_lasso   


# In[53]:


ratio=[0,0.2,0.4,0.6,0.8,1] 
data_ElasticNet=pd.DataFrame(columns=['ratio','Coefs','alpha','RMSE','Adjusted R^2'])

for r in ratio:
    for a in alfa_A:
        ElasticNet_model=ElasticNet(l1_ratio=r ,alpha=a )
        ElasticNet_model.fit(X_train,y_train)
        pred_net = ElasticNet_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test,pred_net))
        r2_e=r2_score(y_test, pred_net)
        n = len(y_train)
        Adj_r2 = 1-(1-r2_e)*(n-1)/(n-p-1)
        data_ElasticNet = data_ElasticNet.append({'ratio': r,'Coefs': ElasticNet_model.coef_,'alpha': a, 'RMSE' : rmse, 'Adjusted R^2' : Adj_r2}, ignore_index = True)


# In[54]:


data_ElasticNet_5=pd.DataFrame(columns=['ratio','Coefs','alpha','RMSE','Adjusted R^2'])

for a in alfa_A:
    ElasticNet_model_5=ElasticNet(l1_ratio=0.5 ,alpha=a )
    ElasticNet_model_5.fit(X_train,y_train)
    pred_net_5 = ElasticNet_model_5.predict(X_test)
    rmse_5 = np.sqrt(mean_squared_error(y_test,pred_net))
    r2_e_5=r2_score(y_test, pred_net)
    n_5 = len(y_train)
    Adj_r2_5 = 1-(1-r2_e_5)*(n_5-1)/(n_5-p-1)
    data_ElasticNet_5 = data_ElasticNet_5.append({'ratio': 0.5 ,'Coefs': ElasticNet_model_5.coef_,'alpha': a, 'RMSE' : rmse_5, 'Adjusted R^2' : Adj_r2_5}, ignore_index = True)


# In[55]:


data_ElasticNet_5


# In[56]:


import matplotlib.pyplot as plt

alpha = [0,0.1,0.25,0.5,1,5]
count = 0
for i in alpha:
    ridge = Ridge(alpha=i)
    ridge.fit(X_train,y_train)
    ElasticNet_model=ElasticNet(l1_ratio=0.5 ,alpha=i )
    ElasticNet_model.fit(X_train,y_train)
    lasso_model=Lasso(alpha=i)
    lasso_model.fit(X_train,y_train)

    r2_score_lasso = r2_score(y_test, pred_laso)
    r2_score_elastic = r2_score(y_test, pred_net)
    r2_score_ridge = r2_score(y_test, pred_ridge)


    m, s, _ = plt.stem(
        np.where(ElasticNet_model_5.coef_)[0],
        ElasticNet_model_5.coef_[ElasticNet_model_5.coef_ != 0],
        markerfmt="o",
        label="Elastic net coefficients",
    )
    plt.setp([m, s], color="m")
    
    m, s, _ = plt.stem(
        np.where(lasso_model.coef_)[0],
        lasso_model.coef_[lasso_model.coef_ != 0],
        markerfmt="x",
        label="Lasso coefficients",
    )
    plt.setp([m, s], color="blue")
    
    m, s, _ = plt.stem(
        np.where(ridge.coef_)[1],
        ridge.coef_[ridge.coef_ != 0],
        markerfmt="x",
        label="Ridge coefficients",
    )

    plt.legend(loc="best")
    plt.title(
        "Lasso $R^2$: %.3f, Elastic Net $R^2$: %.3f, Rigde $R^2$: %.3f" % (r2_score_lasso,
                                                                           r2_score_elastic,
                                                                            r2_score_ridge)
    )

    plt.show()


# In[57]:


x = range(len(data_ElasticNet['Coefs'][0]))
alphas = [0,0.1,0.25,0.5,1,5]

for i in range(len(alphas)):
    alpha = alphas[i]
    
    #Ridge
    markerline, stemlines, baseline = plt.stem(x, data_ridge['Coefs'][i],markerfmt="x", label="Ridge coefficients")
    markerline.set_markerfacecolor('none')
    plt.setp([markerline, stemlines], color="blue")
    
    #Lasso
    markerline, stemlines, baseline = plt.stem(x, data_lasso['Coefs'][i], markerfmt="o", label="Lasso coefficients")
    plt.setp([markerline, stemlines], color="green")

    #EN
    markerline, stemlines, baseline = plt.stem(x, data_ElasticNet['Coefs'][i], markerfmt="d", label="Elastic net coefficients",)
    plt.setp([markerline, stemlines], color="m")

    plt.legend(loc="best")
    plt.title(
    "Rigde $R^2$: %.3f, Lasso $R^2$: %.3f, Elastic Net $R^2$: %.3f" % (data_ridge['Adjusted R^2'][i],
                                                                       data_lasso['Adjusted R^2'][i],
                                                                       data_ElasticNet['Adjusted R^2'][i])
    )
    plt.show()
    


# In[58]:


ridge.coef_[ridge.coef_ != 0]


# In[59]:


ElasticNet_model_5.coef_[ElasticNet_model.coef_ != 0]


# In[ ]:





# In[ ]:




