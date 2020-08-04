#!/usr/bin/env python
# coding: utf-8

# In[150]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("datasets_222487_478477_framingham.csv")


# In[151]:


df.head()


# In[152]:


df.info()


# In[153]:


df.isnull().sum()


# In[154]:


# Since the average and median values are close in most variables, I filled the nan values according to the median.

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
imputer.fit(df.iloc[:,:-1].values)
df.iloc[:,:-1] = imputer.transform(df.iloc[:,:-1].values)


# In[155]:


df.isnull().sum()


# In[156]:


#Outlier Detection (Kuantil)
for column in df.columns[1:-1]:
    for chd in df.TenYearCHD.unique():
        selectedCHD= df[df["TenYearCHD"] == chd]
        selected_column= selectedCHD[column]

        q1=selected_column.quantile(0.25)
        q3=selected_column.quantile(0.75)

        iqr = q3-q1
        minimum = q1-1.5*iqr
        maximum = q3+1.5*iqr

        print(column,chd, "| min= ",minimum, "|max= ",maximum)

        max_idxs = df[(df["TenYearCHD"]== chd) &  (df[column] > maximum)].index
        print(max_idxs)
        min_idxs = df[(df["TenYearCHD"]== chd) &  (df[column] < minimum)].index
        print(min_idxs)
        df.drop(index = max_idxs, inplace = True)
        df.drop(index= min_idxs, inplace = True)


# In[157]:


df.drop(columns="BPMeds",inplace=True)
df.drop(columns="prevalentStroke",inplace=True)
df.drop(columns="diabetes",inplace= True)


# In[158]:


plt.figure(figsize=(20,20))
cor = df.corr()
sns_heat=sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
fig=sns_heat.get_figure
plt.show()


# In[159]:


df.hist(figsize=(20, 20))
plt.show()


# In[160]:


plt.figure(figsize=(10,10))
sns.boxplot(data=df,palette='RdBu',orient='h')


# In[161]:


X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values


# In[162]:


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2,random_state=0)


# In[163]:


from sklearn.preprocessing import StandardScaler
stdc = StandardScaler()
X_train = stdc.fit_transform(X_train)
X_test = stdc.transform(X_test)


# In[164]:


from sklearn.metrics import roc_curve,roc_auc_score,classification_report,accuracy_score

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression().fit(X_train, y_train)

preds = lr.predict(X_test)
print(classification_report(y_test, preds))


# In[165]:



def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
probs = lr.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plot_roc_curve(fpr, tpr)


# In[166]:


cross_val_score(lr,X_test,y_test,cv=10).mean()


# In[167]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier().fit(X_train,y_train)
result = knn.predict(X_test)
print(classification_report(y_test, result))


# In[168]:


#ROC
probs = knn.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plot_roc_curve(fpr, tpr)


# In[169]:


#Knn model tuning
knn_params ={"n_neighbors": np.arange(1,10), "metric": ["euclidean","minkowski","manhattan"]}
knn_cv_model = GridSearchCV(knn,knn_params, cv=10,n_jobs=-1, verbose=2).fit(X_train,y_train)
knn_cv_model.score(X_test,y_test)


# In[170]:


knn_cv_model.best_params_


# In[171]:


knn_tuned = KNeighborsClassifier(metric='euclidean',n_neighbors=9).fit(X_train,y_train)

result = knn_tuned.predict(X_test)

print(classification_report(y_test, result))


# In[172]:


#ROC
probs = knn_tuned.predict_proba(X_test)
probs = probs[:, 1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plot_roc_curve(fpr, tpr)


# In[173]:


from sklearn.svm import SVC
svm_model = SVC().fit(X_train,y_train)
result = svm_model.predict(X_test)
print(classification_report(y_test, result))


# In[174]:


#svm model tuning
svm = SVC() 
svm_params ={"C": [0.1,1,2,3],"kernel": ["rbf","poly"],"degree" : [0, 1, 2,3,4]}
svm_cv_model = GridSearchCV(svm,svm_params,cv=10, n_jobs=-1,verbose=2).fit(X_train,y_train)


# In[175]:


svm_cv_model.best_score_


# In[176]:


svm_cv_model.best_params_


# In[177]:


#final model
svm_tuned = SVC(C=3, kernel= "rbf",degree=0).fit(X_train,y_train)
result = svm_tuned.predict(X_test)
print(classification_report(y_test, result))


# In[178]:


from sklearn.neural_network import MLPClassifier

mlpc_model = MLPClassifier(activation="logistic").fit(X_train,y_train)
result = mlpc_model.predict(X_test)
accuracy_score(y_test,result)


# In[81]:


mlpc_params={"alpha": [0.1,0.01,0.03],"hidden_layer_sizes": [(10,10),(100,100,100),(3,5)],"solver": ["lbfgs","adam"]}


# In[82]:


#mlpc tuning
mlpc=MLPClassifier()
mlpc_cv_model = GridSearchCV(mlpc,mlpc_params ,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)


# In[83]:


mlpc_cv_model.best_score_


# In[84]:


mlpc_cv_model.best_params_


# In[179]:


#mlpc tuned
mlpc_tuned = MLPClassifier(activation="logistic",alpha=0.1,hidden_layer_sizes=(10,10),solver="adam").fit(X_train,y_train)
result = mlpc_tuned.predict(X_test)
accuracy_score(y_test,result)


# In[180]:


from sklearn.tree import DecisionTreeClassifier
cart_model = DecisionTreeClassifier().fit(X_train,y_train)


# In[181]:


result = cart_model.predict(X_test)
accuracy_score(y_test,result)


# In[88]:


cart = DecisionTreeClassifier()
cart_params={"max_depth": [1,3,5,7],"min_samples_split": [2,3,6,10,15,20]}


# In[89]:


cart_cv_model = GridSearchCV(cart,cart_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)


# In[90]:


cart_cv_model.best_params_


# In[182]:


cart_tuned = DecisionTreeClassifier(max_depth=1,min_samples_split=2).fit(X_train,y_train)


# In[183]:


result = cart_tuned.predict(X_test)
accuracy_score(y_test,result)


# In[184]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
rf_model = RandomForestClassifier().fit(X_train,y_train)


# In[185]:


result = rf_model.predict(X_test)
accuracy_score(y_test,result)


# In[42]:


#rf model tuning
rf = RandomForestClassifier()
rf_params = {"n_estimators": [100,200,500,1000],
            "max_features": [1,3,5,7],
            "min_samples_split": [2,5,7]}
rf_cv_model = GridSearchCV(rf,rf_params,cv=10, n_jobs=-1,verbose=2).fit(X_train,y_train)


# In[43]:


rf_cv_model.best_params_


# In[186]:


rf_tuned = RandomForestClassifier(max_features= 1,min_samples_split= 2,n_estimators=1000).fit(X_train,y_train)
result = rf_tuned.predict(X_test)
accuracy_score(y_test,result)


# In[187]:


from sklearn.ensemble import GradientBoostingClassifier
gbm_model = GradientBoostingClassifier().fit(X_train,y_train)
gbm_model


# In[188]:


result = gbm_model.predict(X_test)
accuracy_score(y_test,result)


# In[195]:


gbm = GradientBoostingClassifier()
gbm_params = {"learning_rate": [0.1,0.001,0.05],
              "n_estimators": [100,500,1000],
              "max_depth": [2,3,5,8]}

gbm_cv_model= GridSearchCV(gbm,gbm_params,cv=10, n_jobs=-1,verbose=2).fit(X_train,y_train)


# In[194]:


gbm_cv_model.best_params_


# In[190]:


gbm_tuned = GradientBoostingClassifier(learning_rate=0.001,max_depth=2,n_estimators=100).fit(X_train,y_train)
result = gbm_tuned.predict(X_test)
accuracy_score(y_test,result)


# In[191]:


from xgboost import XGBClassifier
xgb_model = XGBClassifier(objective="binary:logistic").fit(X_train,y_train)


# In[192]:


result = xgb_model.predict(X_test)
accuracy_score(y_test,result)


# In[52]:


xgb = XGBClassifier()
xgb_params = {"learning_rate": [0.1,0.001,0.05],
              "n_estimators": [100,500,1000,2000],
              "max_depth": [2,3,5,8]}


# In[53]:


xgb_cv_model = GridSearchCV(xgb,xgb_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)


# In[54]:


xgb_cv_model.best_params_


# In[55]:


xgb_tuned = XGBClassifier(learning_rate= 0.1,max_depth= 2,n_estimators= 100).fit(X_train,y_train)


# In[56]:


result = xgb_tuned.predict(X_test)
accuracy_score(y_test,result)


# In[57]:


from lightgbm import LGBMClassifier


# In[58]:


lgbm_model = LGBMClassifier().fit(X_train,y_train)
lgbm_model


# In[59]:


result = lgbm_model.predict(X_test)
accuracy_score(y_test,result)


# In[60]:


lgbm = LGBMClassifier()
lgbm_params= {"learning_rate": [0.1,0.001,0.05],
              "n_estimators": [100,500,1000],
              "max_depth": [2,3,5,8]}
lgbm_cv_model = GridSearchCV(lgbm,lgbm_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)


# In[61]:


result = lgbm_model.predict(X_test)
accuracy_score(y_test,result)


# In[62]:


lgbm_cv_model.best_params_


# In[63]:


lgbm_tuned = LGBMClassifier(learning_rate=0.1,max_depth=3,n_estimators=100).fit(X_train,y_train)
result = lgbm_model.predict(X_test)
accuracy_score(y_test,result)


# In[ ]:




