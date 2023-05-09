# -*- coding: utf-8 -*-
"""All_Data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Xe0HDhkxqeR4gEw0Si1-c_R3mLbT3WSd
"""


# Step 2: Load the package


import numpy as np
 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.decomposition import PCA

from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastSurvivalSVM
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastSurvivalSVM
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxnetSurvivalAnalysis
#from lifelines import CoxPHFitter 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib 
matplotlib.rcParams['pdf.fonttype'] = 42 
matplotlib.rcParams['ps.fonttype'] = 42

# Step 2: Load data

# Load data
RNA_32=pd.read_csv('C:\\Users\\Noushin\\OneDrive - Teesside University\\Noushin-paper\\Feture_vector\\32-size\\patient_RNA_62.csv')
Meta_32=pd.read_csv('C:\\Users\\Noushin\\OneDrive - Teesside University\\Noushin-paper\\Feture_vector\\32-size\\patient_Meta_59.csv')
IMG_32=pd.read_csv('C:\\Users\\Noushin\\OneDrive - Teesside University\\Noushin-paper\\Feture_vector\\32-size\\patient_IMG_83.csv')
feature1= pd.merge(IMG_32,RNA_32,on='sample')
data= pd.merge(feature1,Meta_32,on='sample')
#RNA_coxCC_2 = pd.read_csv('/content/drive/MyDrive/Feature-vector/RNA_2_16.csv')
#Feature_vector= pd.merge(RNA_coxCC,RNA_coxCC_2 ,on='sample')
Survival_Data = pd.read_excel('D:\\Second-year\\survival-data\\survival.xlsx', sheet_name='TCGA-CDR', usecols="A,Y, Z")
Feature_vector_Final= pd.merge(data,Survival_Data,on='sample')

# 3.1 Merge gene data with OS time and status
Feature_vector_Final = Feature_vector_Final.rename(columns={ 'sample':'bcr_patient_barcode'})
Feature_vector_Final = Feature_vector_Final.rename(columns={ 'OS':'OS.Status'})

from sklearn.preprocessing import OrdinalEncoder
data = Feature_vector_Final.drop(['bcr_patient_barcode','OS.Status','OS.time'], axis=1)

data.head()

#preprocess then add the label to df again/ different preprocessing method you can use
#ss = MinMaxScaler()

#df = data
#df = pd.DataFrame(ss.fit_transform(df), columns=df.columns)
#df['OS.Status'] = Feature_vector_Final['OS.Status']
#df['OS.time'] = Feature_vector_Final['OS.time']

# second method for preprocessing /compare both
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

df = data
df = pd.DataFrame(ss.fit_transform(df), columns=df.columns)
df['OS.Status'] = Feature_vector_Final['OS.Status']
df['OS.time'] = Feature_vector_Final['OS.time']

# Step 6: Machine Learning Methods for Survival Analysis

# 6.1: Set up seed
SEED = 50
CV = KFold(n_splits=3, shuffle=True, random_state=0)
TEST_SIZE =0.3

# 6.2 Split data to prepare for ML (already normalise)

X = df.drop(['OS.time','OS.Status'], axis = 1)
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.6 * (1 - .8)))
sel.fit_transform(X)
df['OS.Status'] = np.where(df['OS.Status'] == 1, True, False)
y = df[['OS.Status','OS.time']].to_records(index=False)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

#X_test.index

# Define a function for grid search to tune training model
# and predict the results
def grid_search(estimator, param, X_train, y_train, X_test, y_test, CV):
    
    # Define Grid Search
    gcv = GridSearchCV(
    estimator,
    param_grid=param,
    cv=CV,
    n_jobs=-1).fit(X_train, y_train)

    # Find best model
    model = gcv.best_estimator_
    print(model)


    # Predict model
    prediction = model.predict(X_test)
    result = concordance_index_censored(y_test["OS.Status"], y_test["OS.time"], prediction)
    print('C-index for test set (Hold out):', result[0])

    return [model, round(result[0],3), prediction]

# 6.3: Build model

# Define a function for grid search to tune training model
# and predict the results
def c_index(model, X, y, n=5):
      
    # Generate seed 
    np.random.seed(1)
    seeds = np.random.permutation(1000)[:n]

    # Train and evaluate model with 20 times
    cindex_score = []
    predict_list = []
    
    for s in seeds:
   
        #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=10)
        X_trn, X_test, y_trn, y_test = train_test_split(X, y, test_size=0.3,random_state=s)
        model.fit(X_trn, y_trn)
        # cv_score = cross_val_score(model, X_test, y_test, cv=CV)
        prediction = model.predict(X_test)
        predict_list.append(prediction)
        result = concordance_index_censored(y_test["OS.Status"],y_test["OS.time"], prediction)
    
        cindex_score.append(round(result[0],3))

    print('Average C-index Cross validation:', np.mean(cindex_score))
   

    return [cindex_score, predict_list]

"""# Gradient Boost Survival
pipe_gbs = Pipeline([('model', GradientBoostingSurvivalAnalysis())])
param_gbs ={
        'model__random_state': [SEED],
        'model__learning_rate': [0.01, 0.1, 1],
        'model__n_estimators':[100,200, 500, 800, 1000],
        'model__min_samples_split':[2,10,8],
        'model__min_samples_leaf':[1,2,8],
        'model__max_depth':[3,4,8]}
"""

# Define the Pipeline and hyperparameter
# Random Survival Forest
pipe_rsf = Pipeline([('model', RandomSurvivalForest())])
param_rsf ={
        'model__random_state': [SEED],
        'model__max_features': ['sqrt','auto'],
        'model__max_depth': [8,16,32,64],
        'model__min_samples_leaf': [8,15],
        'model__n_estimators':[200,500,800, 1000]}



# ComponentwiseGradientBoostingSurvivalAnalysis
pipe_cgbs = Pipeline([('model', ComponentwiseGradientBoostingSurvivalAnalysis())])
param_cgbs ={
        'model__random_state': [50,20,200],
        'model__learning_rate': [0.001,0.01, 0.1, 1],
        'model__n_estimators':[200, 500, 800, 1000],}



# Gradient Boost Survival
pipe_gbs = Pipeline([('model', GradientBoostingSurvivalAnalysis())])
param_gbs ={
        'model__random_state': [SEED],
        'model__learning_rate': [0.01, 0.1, 1],
        'model__n_estimators':[200, 500, 800, 1000]}


# CoxPHSurvivalAnalysis
pipe_cox = Pipeline([('model', CoxPHSurvivalAnalysis())])
param_cox ={
        "model__alpha": [0.001, 0.01, 0.1, 0, 1, 10, 100]}
      
# CoxnetSurvivalAnalysis
pipe_coxnet = Pipeline([('model', CoxnetSurvivalAnalysis())])
param_coxnet ={
        "model__alpha_min_ratio": [0.001, 0.01,0.5, 0.1],
        "model__l1_ratio": [0.0001,0.001, 0.01, 0.5,0.9,1]}




# Survival SVM
pipe_svm = Pipeline([('model', FastSurvivalSVM())])
param_svm ={
       'model__random_state': [SEED],
       'model__max_iter': [20, 100, 1000,100000],
       'model__optimizer':['avltree', 'rbtree',]}

# Estimator list:
#'Cox Regression':[pipe_cox, param_cox ], 
estimator_list = {
                  'Random Forest Survival':[pipe_rsf, param_rsf],
                  'Component Wise Gradient Boosting Survival': [pipe_cgbs, param_cgbs],
                  'Gradient Boosting Survival': [pipe_gbs, param_gbs], 
                  'Cox Regression':[pipe_cox, param_cox ], 
                  'Coxnet Regression':[pipe_coxnet, param_coxnet ],
                 'SVM Survival': [pipe_svm, param_svm]}

estimator_list.items()



#%%
#Nested CV

hold_out_results = []
model_list = []
pred_list = []
c_index_list = []
pred_list_n = []
pre_list_CV=[]

for model_name, index in estimator_list.items():
  print('\n',model_name)
  for train_ix, test_ix in CV.split(X):
    X_train, X_test = X.loc[train_ix, :], X.loc[test_ix, :]
    y_train, y_test = y[train_ix],y[test_ix]
        #X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=10)
    
        #print('\n',model_name)
    estimator = index[0]
    param = index[1]
    outcome = grid_search(estimator, param, X_train, y_train, X_test, y_test, CV)
    model = outcome[0]
    hold_out_results.append(outcome[1])
    pred_list.append(outcome[2])
  
    
        # Run model n times to check c-index
    score, pre = c_index(model, X, y, n=5)
    c_index_list.append(score)
    pred_list_n.append(pre)
  pre_list_CV.append(c_index_list)
#%%
#Nested CV c-index 

print('RSF mean:',np.mean(c_index_list[0:3]) )
print('CGBSurv mean:',np.mean(c_index_list[4:6]) )
print('GBSurv mean:',np.mean(c_index_list[6:9]) )
print('CoxPH mean:',np.mean(c_index_list[9:12]) )
print('Coxnet mean:',np.mean(c_index_list[12:15]) )
print('SurvSVM mean:',np.mean(c_index_list[15:]) )

#%%

#nested vertion box plot
my_array=[]
my_array = np.array(c_index_list)
a= np.concatenate([my_array[0,:],my_array[1,:],my_array[2,:]])
b=np.concatenate([my_array[3,:],my_array[4,:],my_array[5,:]])
c=np.concatenate([my_array[6,:],my_array[7,:],my_array[8,:]])
d=np.concatenate([my_array[9,:],my_array[10,:],my_array[11,:]])
e=np.concatenate([my_array[12,:],my_array[13,:],my_array[14,:]])
g= np.concatenate([my_array[15,:],my_array[16,:],my_array[17,:]])
cc=[a,b,c,d,e,g]


name = ['RSF','CGBSurv', 'GBSurv','CoxPH','Coxnet', 'SurvSVM']
res = []
#'GBSurv',
for i in range(0,6):
    for c in cc[i]:
        res.append([name[i],c])
       
c_plot = pd.DataFrame(res, columns=['Learning approach','C-Index'])
ax=sns.set_style("white")
#ax = sns.boxplot(x="Learning approach", y="C-Index", data=c_plot)
ax = sns.boxplot(x="Learning approach", y="C-Index", data=c_plot)
ax = sns.stripplot(x="Learning approach", y="C-Index", data=c_plot, jitter=False, edgecolor="black")
# A violin plot combines the benefits of the previous two plots and simplifies them
# Denser regions of the data are fatter, and sparser thiner in a violin plot
sns.violinplot(x="Learning approach", y="C-Index", data=c_plot, size=6, edgecolor="black")
#plt.title('All Data')
#plt.title('Images and Transcriptomics Data')
plt.title('Metabolomics and Transcriptomics Data')
#plt.title('Metabolomics and Images Data')
ax=sns.set_style("white")
sns.set_style("whitegrid", {'axes.grid' : False})
plt.ylim(0.3, 1)
plt.savefig('C:\\Users\\Noushin\\OneDrive - Teesside University\\Noushin-paper\\result\\New_pdf_adobe\\Meta_RNA_plot_nested_1.pdf',transparent=True)

#%%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
#%%                                              
hold_out_results = []
model_list = []
pred_list = []
c_index_list = []
pred_list_n = []

for model_name, index in estimator_list.items():
    print('\n',model_name)
    estimator = index[0]
    param = index[1]
    outcome = grid_search(estimator, param, X_train, y_train, X_test, y_test, CV)
    model = outcome[0]
    hold_out_results.append(outcome[1])
    pred_list.append(outcome[2])

    # Run model n times to check c-index
    score, pre = c_index(model, X, y, n=15)
    c_index_list.append(score)
    pred_list_n.append(pre)


#%%
# Visual results

name = ['RSF','CGBSurv', 'GBSurv','CoxPH','Coxnet', 'SurvSVM']
cv_res = []
#'GBSurv',
for i in range(0,6):
    for c in c_index_list[i]:
        cv_res.append([name[i],c])

c_plot = pd.DataFrame(cv_res, columns=['Learning approach','C-Index'])
ax=sns.set_style("white")
#ax = sns.boxplot(x="Learning approach", y="C-Index", data=c_plot)
#ax = sns.boxplot(x="Learning approach", y="C-Index", data=c_plot)
ax = sns.stripplot(x="Learning approach", y="C-Index", data=c_plot, jitter=True, edgecolor="black")
# A violin plot combines the benefits of the previous two plots and simplifies them
# Denser regions of the data are fatter, and sparser thiner in a violin plot
sns.violinplot(x="Learning approach", y="C-Index", data=c_plot, size=6, edgecolor="black")
plt.title('All Data')
#plt.title('Images and Transcriptomics Data')
#plt.title('Metabolomics and Transcriptomics Data')
#plt.title('Metabolomics and Images Data')
ax=sns.set_style("white")
sns.set_style("whitegrid", {'axes.grid' : False})
plt.ylim(0.3, 1)
#plt.savefig('C:\\Users\\Noushin\\OneDrive - Teesside University\\AllData_plot.pdf',transparent=True)






#%%
CoxPH = c_plot[c_plot['Learning approach'] == 'CoxPH']['C-Index'].values
Coxnet = c_plot[c_plot['Learning approach'] == 'Coxnet']['C-Index'].values
RSF = c_plot[c_plot['Learning approach'] == 'RSF']['C-Index'].values
GBSurv = c_plot[c_plot['Learning approach'] == 'GBSurv']['C-Index'].values
CGBSurv = c_plot[c_plot['Learning approach'] == 'CGBSurv']['C-Index'].values
SurvSVM = c_plot[c_plot['Learning approach'] == 'SurvSVM']['C-Index'].values

print('CoxPH median:',np.median(CoxPH) )
print('Coxnet median:',np.median(Coxnet) )
print('RSF median:',np.median(RSF) )
print('GBSurv median:',np.median(GBSurv) )
print('CGBSurv median:',np.median(CGBSurv) )
print('SurvSVM median:',np.median(SurvSVM) )
#%%
from scipy import stats
#t, p = stats.ttest_ind(CGBSurv, GBSurv)
#t, p = stats.ttest_ind(CoxPH, Coxnet)
#t, p = stats.ttest_ind(Coxnet, SurvSVM)
#t, p = stats.ttest_ind(Coxnet, RSF)
t, p = stats.ttest_ind(RSF, CGBSurv)
#t, p = stats.ttest_ind(GBSurv, CoxPH)
#t, p = stats.ttest_ind(Coxnet , CGBSurv)
#t, p = stats.ttest_ind(CoxPH , RSF)

#%%
"""Interpretation:
If the t-value is positive (>0) then the mean of g1  was significantly greater than the mean of g2.


If the t-value is negative (<0) then the mean of g1 was significantly smaller than the mean of g2 .
"""



from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter 
from lifelines.plotting import add_at_risk_counts

names = list(estimator_list.keys())
k=0
fig, ax = plt.subplots(2,3, figsize=(12,12))
# fig, ax = plt.subplots()
for pred in pred_list:
    df1 = X_test.reset_index(drop=True) 
    y_pred = pred
    med = np.median(y_pred)
   
    r = np.where(y_pred >= med, 1, 0)
    df1['Risk'] = r
    df1['pred'] = pred
    df3=df1.sort_values(by='pred', ascending=False)
    High_risk=df3.head(n=6)
    Low_risk=df3.tail(n=6)
    print(df1.shape)
    ix = df1['Risk'] == 1
    #ix=High_risk.index
    #ix1=Low_risk.index

    df_y = pd.DataFrame(y_test)
    df_y['OS.Status'] = np.where(df_y['OS.Status'] == True, 1, 0)
    df1['OS.Status']= df_y['OS.Status']
    df1['OS.time']= df_y['OS.time']
    T_hr, E_hr = df1.loc[ix]['OS.time'], df1.loc[ix]['OS.Status']
    T_lr, E_lr = df1.loc[~ix]['OS.time'], df1.loc[~ix]['OS.Status']
    #T_lr, E_lr = df1.loc[ix1]['OS.time'], df1.loc[ix1]['OS.Status']


    # Set-up plots
    k +=1
    plt.subplot(3,2,k)

    # Fit survival curves
    kmf_control = KaplanMeierFitter()
    ax = kmf_control.fit(T_hr, E_hr, label='HR').plot_survival_function()

    kmf_exp = KaplanMeierFitter()
    ax = kmf_exp.fit(T_lr, E_lr, label='LR').plot_survival_function()
    

    add_at_risk_counts(kmf_exp, kmf_control)
    # Format graph
    plt.ylim(0,1)
    ax.set_xlabel('Timeline (months)',fontsize='large')
    ax.set_ylabel('Percentage of Population Alive',fontsize='large')

    # Calculate p-value
    # res = logrank_test(T_hr, T_lr, alpha=.95)
    res = logrank_test(T_hr, T_lr, event_observed_A=E_hr, event_observed_B=E_lr, alpha=.95)
    print('\nModel', names[k-1])
    res.print_summary()
  

    # Location the label at the 1st out of 9 tick marks
    xloc = max(np.max(T_hr),np.max(T_lr)) / 10
    ax.text(xloc,.2,res.p_value,fontsize=12)
    plt.subplot(3,2,k).set_title('KM Curves {}' .format(names[k-1]))
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.tight_layout()
    #plt.savefig('D:\\Second-year\\Result\\All Data_km_30%.pdf')
    
    
#HR=pred_list
#np.savetxt("D:\\Second-year\\Result\\GFG1.csv", pred_list,delimiter =", ", fmt ='% s')
#HR_index=pd.read_csv("D:\\Second-year\\Result\\GFG1.csv")
#HR_index=pd.read_csv("D:\\Second-year\\Result\\GFG1.csv",header=None)
#HR_index_t=HR_index.T
#np.savetxt("D:\\Second-year\\Result\\GFG_t.csv", HR_index_t,delimiter =", ", fmt ='% s')
#print(np.median(pred_list[2]))
#df.to_csv('D:\\Second-year\\Result\\index-data.csv')
#%%

from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter 
from lifelines.plotting import add_at_risk_counts

names = list(estimator_list.keys())
k=0
#fig, ax = plt.subplots(2,1, figsize=(10,12))
fig, ax = plt.subplots(3,2, figsize=(12,12))
# fig, ax = plt.subplots()
for pred in pred_list:
    df1 = X_test.reset_index(drop=True) 
    y_pred = pred
    med = np.median(y_pred)
    r = np.where(y_pred >= med, 1, 0)
    #df1['Risk'] = r
    df1['pred'] = pred
    print(df1.shape)
    df3=df1.sort_values(by='pred', ascending=False)
    High_risk=df3.head(n=10)
    Low_risk=df3.tail(n=10)
    #ix = df1['Risk'] == 1
    ix=High_risk.index
    ix1=Low_risk.index

    df_y = pd.DataFrame(y_test)
    df_y['OS.Status'] = np.where(df_y['OS.Status'] == True, 1, 0)
    df1['OS.Status']= df_y['OS.Status']
    df1['OS.time']= df_y['OS.time']
    T_hr, E_hr = df1.loc[ix]['OS.time'], df1.loc[ix]['OS.Status']
    #T_lr, E_lr = df1.loc[~ix]['OS.time'], df1.loc[~ix]['OS.Status']
    T_lr, E_lr = df1.loc[ix1]['OS.time'], df1.loc[ix1]['OS.Status']


    # Set-up plots
    
    k +=1
    #plt.subplot(2,1,k)
    plt.subplot(3,2,k)

    # Fit survival curves
    kmf_control = KaplanMeierFitter()
    ax = kmf_control.fit(T_hr, E_hr, label='HR').plot_survival_function()

    kmf_exp = KaplanMeierFitter()
    ax = kmf_exp.fit(T_lr, E_lr, label='LR').plot_survival_function()
    

    add_at_risk_counts(kmf_exp, kmf_control)
    # Format graph
    plt.ylim(0,1)
    ax.set_xlabel('Timeline (months)',fontsize='large')
    ax.set_ylabel('Percentage of Population Alive',fontsize='large')

    # Calculate p-value
    # res = logrank_test(T_hr, T_lr, alpha=.95)
    res = logrank_test(T_hr, T_lr, event_observed_A=E_hr, event_observed_B=E_lr, alpha=.95)
    print('\nModel', names[k-1])
    res.print_summary()
  

    # Location the label at the 1st out of 9 tick marks
    xloc = max(np.max(T_hr),np.max(T_lr)) / 10
   
    
    ax.text(xloc,.2,r'$\mathrm{p}=%.3f$' % (res.p_value, ),fontsize=18)
    plt.subplot(3,2,k).set_title('{}' .format(names[k-1]))
    #plt.subplot(2,1,k).set_title('{}' .format(names[k-1]))
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['font.size'] = 14
    #plt.rcParams['font.family'] = 'Arial'
    #plt.rcParams.update({'font.family':'Arial'})
    #plt.rcParams['mathtext.rm'] = 'Arial'
    from matplotlib.pyplot import *

# Need to use precise font names in rcParams; I found my fonts with
    import matplotlib.font_manager
    [f.name for f in matplotlib.font_manager.fontManager.ttflist]

    rcParams['mathtext.fontset'] = 'custom'
    rcParams['mathtext.it'] = 'Arial:italic'
    rcParams['mathtext.rm'] = 'Arial'
    plt.tight_layout()
    
    #plt.savefig('D:\\Second-year\\Result\\New\\img_rna_meta_two_3.pdf',transparent=True)
    
#%%

import sklearn
import shap
# use Kernel SHAP to explain test set predictions
X100 = shap.utils.sample(X_test, 17) # 100 instances for use as the background distribution
# compute the SHAP values for the linear model
pipe_coxnet.fit(X_train,y_train)
explainer = shap.Explainer(pipe_coxnet.predict, X100)
shap_values = explainer(X_test)

# make a standard partial dependence plot
sample_ind = 17
# the waterfall_plot shows how we get from shap_values.base_values to model.predict(X)[sample_ind]
#shap.plots.waterfall(shap_values[sample_ind], max_display=20)
#plt.savefig('D:\\Second-year\\Result\\New\\shap.pdf',transparent=True)
shap.plots.force(shap_values[0])
#%%

import sklearn
import shap
# use Kernel SHAP to explain test set predictions
X100 = shap.utils.sample(X_test, 20) # 100 instances for use as the background distribution
# compute the SHAP values for the linear model
pipe_coxnet.fit(X_train,y_train)
explainer = shap.Explainer(pipe_coxnet.predict, X100)
shap_values = explainer(X_test)

# make a standard partial dependence plot
sample_ind = 16
# the waterfall_plot shows how we get from shap_values.base_values to model.predict(X)[sample_ind]
shap.plots.waterfall(shap_values[sample_ind], max_display=30)
plt.savefig('D:\\Second-year\\Result\\New\\shap.pdf',transparent=True)