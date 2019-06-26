#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
pd.options.display.max_rows=10
from datetime import datetime
from statsmodels.tsa.arima_model import ARIMA
import xgboost
import graphviz
import matplotlib.dates as md
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import scipy
from scipy.stats import spearmanr
from pylab import rcParams
import seaborn as sb
from matplotlib.pylab import rcParams
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import MyAnalysis 
rcParams['figure.figsize'] = 3,3

#Import for threading

import random
import sys
from threading import Thread, RLock
import time



import numpy as np
df = pd.read_table("grenoble_7.k7",sep = ',',header = 0)

def trace_conf_mat(cm, acc,classes, norm, title,cmap=plt.cm.Blues):
    accuracy=acc
    if norm:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title+'%.2f' % accuracy,
           ylabel='True label',
           xlabel='Predicted label')
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
     #        rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if norm else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def plot_confusion_matrix_pdr(df, classes, cm,accuracy, normalize=False, title=None,cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    trace_conf_mat(cm, accuracy,classes, normalize, title,cmap=plt.cm.Blues)
np.set_printoptions(precision=2)


def my_split(df,src,dst,size_of_test,kind):
    ros = RandomOverSampler(random_state=0)
    rus = RandomUnderSampler(random_state=0)
    target=list()
    target_names=list()
    
    if kind=="all":                                              #it is for pdr+rssi
        combined_features=list()
        for i in range(len(df)):
            combined_features.append(df['pdr'][i]*df['pdr'][i]+df['mean_rssi'][i]*df['mean_rssi'][i])
            if df['pdr'][i] <= 0.3:
                target.append(0)
                target_names.append('Bad')
            elif df['pdr'][i]>=0.75:
                target.append(2)
                target_names.append('Good')
            elif ((df['pdr'][i]>0.3 and df['pdr'][i]<0.75)and df['mean_rssi'][i]<=-87):
                target.append(0)
                target_names.append('Bad')
            else:
                target.append(1)
                target_names.append('Interm.')

        df['combined_features']=combined_features
        colonne=['combined_features']
    elif kind=="pdr":                                            #It is for pdr
        for i in range(len(df)):
            if df['pdr'][i] <= 0.3 :
                target.append(0)
                target_names.append('Bad')
            elif df['pdr'][i]>=0.75 :
                target.append(2)
                target_names.append('Good')
            else:
                target.append(1)
                target_names.append('Interm.')
            colonne=['pdr']
    else:                                                          #It is for mean_rssi
        for i in range(len(df)):
            if df['mean_rssi'][i] <= -87 :
                target.append(0)
                target_names.append('Bad')
            elif df['mean_rssi'][i]>=-85 :
                target.append(2)
                target_names.append('Good')
            else:
                target.append(1)
                target_names.append('Interm.')
        colonne=['mean_rssi']
    df['target']=target
    df['target_names']=target_names
    colonne2=['target_names']
    df=df.loc[(df['src']==src)&(df['dst']==dst)]
    
    return df


#Logistic regression Classification
def my_logreg(df1,src,dst,size_of_test,kind,path):#Kind can be "pdr", "mean_rssi" or "all"
    df=my_split(df1,src,dst,size_of_test,kind)
    text=""
    subtitle="Accuracy = "
    if kind=="all":
        title=path+"LR\\Log_Reg_CM_ALL_"+str(src)+"==="+str(dst)+".png"
        title2=path+"LR\\Log_Reg_CM_Not_Normalized_ALL_"+str(src)+"==="+str(dst)+".png"
        colonne=['combined_features']
    elif kind=="pdr":
        title=path+"LR\\Log_Reg_CM_PDR_"+str(src)+"==="+str(dst)+".png"
        title2=path+"LR\\Log_Reg_CM_Not_Normalized_PDR_"+str(src)+"==="+str(dst)+".png"
        colonne=['pdr']
    else:
        title=path+"LR\\Log_Reg_CM_RSSI_"+str(src)+"==="+str(dst)+".png"
        title2=path+"LR\\Log_Reg_CM_Not_Normalized_RSSI_"+str(src)+"==="+str(dst)+".png"
        colonne=['mean_rssi']

    general_test_labels=list()
    general_pred_labels=list()
    channel_list=df1['channel'].unique()
    classes=df['target_names'].unique()
    accuracy_results=list()
    for i in range (len(channel_list)):
        channel_i=channel_list[i]
        colonne2=['target_names']
        ts=df.loc[(df['channel']==channel_i),colonne]
        ts2=df.loc[(df['channel']==channel_i),colonne2]

        features=ts.values
        labels=ts2['target_names']
        #classes=labels.unique()
        #if len(classes)>1:
         #   features, labels=rus.fit_resample(features, labels)#The ressampling strategy
         #   labels=pd.Series(labels)
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = size_of_test, random_state=None,shuffle=True)
        taill=len(pd.Series(train_labels).unique())
        if taill>1:
            LogReg=LogisticRegression()
            LogReg.fit(train_features,train_labels)
            pred_labels=LogReg.predict(test_features)
            general_test_labels=general_test_labels+list(test_labels)
            general_pred_labels=general_pred_labels+list(pred_labels)
        else:
            general_test_labels=general_test_labels+list(test_labels)
            general_pred_labels=general_pred_labels+list(test_labels)
    cm=confusion_matrix(general_test_labels,general_pred_labels)
    som=0
    taill=len(cm)
    total=0
    for i in range(taill):
        som=som+cm[i][i]
        for j in range(taill):
            total=total+cm[i][j]
        
    accuracy=som/total
    plot_confusion_matrix_pdr(df, classes,cm,accuracy, normalize=True,title=subtitle)
    
    plt.savefig(title, format='png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    
    plot_confusion_matrix_pdr(df, classes,cm,accuracy, normalize=False,title=subtitle)
    plt.savefig(title2, format='png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    return accuracy
    
    
#SVM Classification
def my_svm(df1,src,dst,size_of_test,kind,path):#Kind can be "pdr", "mean_rssi" or "all"
    df=my_split(df1,src,dst,size_of_test,kind)
    subtitle="Accuracy = "
    if kind=="all":
            title=path+"SVM\\SVM_CM_ALL_"+str(src)+"==="+str(dst)+".png"
            title2=path+"SVM\\SVM_CM_Not_Normalized_ALL_"+str(src)+"==="+str(dst)+".png"
            colonne=['combined_features']
    elif kind=="pdr":
        title=path+"SVM\\SVM_CM_PDR_"+str(src)+"==="+str(dst)+".png"
        title2=path+"SVM\\SVM_CM_Not_Normalized_PDR_"+str(src)+"==="+str(dst)+".png"
        colonne=['pdr']
    else:
        title=path+"SVM\\SVM_CM_RSSI_"+str(src)+"==="+str(dst)+".png"
        title2=path+"SVM\\SVM_CM_Not_Normalized_RSSI_"+str(src)+"==="+str(dst)+".png"
        colonne=['mean_rssi']

    general_test_labels=list()
    general_pred_labels=list()
    channel_list=df1['channel'].unique()
    classes=df['target_names'].unique()
    accuracy_results=list()
    for i in range (len(channel_list)):
        channel_i=channel_list[i]
        colonne2=['target_names']
        ts=df.loc[(df['channel']==channel_i),colonne]
        ts2=df.loc[(df['channel']==channel_i),colonne2]

        features=ts.values
        labels=ts2['target_names']
        #classes=labels.unique()
        #if len(classes)>1:
        #    features, labels=rus.fit_resample(features, labels)#The ressampling strategy
        #    labels=pd.Series(labels)
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = size_of_test, random_state=None,shuffle=True)
        taill=len(pd.Series(train_labels).unique())
        if taill>1:
            clf = SVC(random_state=0, tol=1e-5)
            clf.fit(train_features, train_labels)
            pred_labels=clf.predict(test_features)
            general_test_labels=general_test_labels+list(test_labels)
            general_pred_labels=general_pred_labels+list(pred_labels)
        else:
            general_test_labels=general_test_labels+list(test_labels)
            general_pred_labels=general_pred_labels+list(test_labels)
    cm=confusion_matrix(general_test_labels,general_pred_labels)
    som=0
    taill=len(cm)
    total=0
    for i in range(taill):
        som=som+cm[i][i]
        for j in range(taill):
            total=total+cm[i][j]
        
    accuracy=som/total
    plot_confusion_matrix_pdr(df, classes,cm,accuracy, normalize=True,title=subtitle)
    
    plt.savefig(title, format='png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    
    plot_confusion_matrix_pdr(df, classes,cm,accuracy, normalize=False,title=subtitle)
    plt.savefig(title2, format='png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    return accuracy
    
#Linear SVM Classification
def my_linear_svm(df1,src,dst,size_of_test,kind,path):#Kind can be "pdr", "mean_rssi" or "all"
    df=my_split(df1,src,dst,size_of_test,kind)
    subtitle="Accuracy = "
    if kind=="all":
        title=path+"LSVM\\LSVM_CM_ALL_"+str(src)+"==="+str(dst)+".png"
        title2=path+"LSVM\\LSVM_CM_Not_Normalized_ALL_"+str(src)+"==="+str(dst)+".png"
        colonne=['combined_features']
    elif kind=="pdr":
        title=path+"LSVM\\LSVM_CM_PDR_"+str(src)+"==="+str(dst)+".png"
        title2=path+"LSVM\\LSVM_CM_Not_Normalized_PDR_"+str(src)+"==="+str(dst)+".png"
        colonne=['pdr']
    else:
        title=path+"LSVM\\LSVM_CM_RSSI_"+str(src)+"==="+str(dst)+".png"
        title2=path+"LSVM\\LSVM_CM_Not_Normalized_RSSI_"+str(src)+"==="+str(dst)+".png"
        colonne=['mean_rssi']

    general_test_labels=list()
    general_pred_labels=list()
    channel_list=df1['channel'].unique()
    classes=df['target_names'].unique()
    accuracy_results=list()
    for i in range (len(channel_list)):
        channel_i=channel_list[i]
        colonne2=['target_names']
        ts=df.loc[(df['channel']==channel_i),colonne]
        ts2=df.loc[(df['channel']==channel_i),colonne2]

        features=ts.values
        labels=ts2['target_names']
        #classes=labels.unique()
        #if len(classes)>1:
        #    features, labels=rus.fit_resample(features, labels)#The ressampling strategy
         #   labels=pd.Series(labels)
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = size_of_test, random_state=None,shuffle=True)
        taill=len(pd.Series(train_labels).unique())
        if taill>1:
            clf = LinearSVC(random_state=0, tol=1e-5)
            clf.fit(train_features, train_labels)
            pred_labels=clf.predict(test_features)
            general_test_labels=general_test_labels+list(test_labels)
            general_pred_labels=general_pred_labels+list(pred_labels)
        else:
            general_test_labels=general_test_labels+list(test_labels)
            general_pred_labels=general_pred_labels+list(test_labels)
    cm=confusion_matrix(general_test_labels,general_pred_labels)
    som=0
    taill=len(cm)
    total=0
    for i in range(taill):
        som=som+cm[i][i]
        for j in range(taill):
            total=total+cm[i][j]
        
    accuracy=som/total
    plot_confusion_matrix_pdr(df, classes,cm,accuracy, normalize=True,title=subtitle)
    
    plt.savefig(title, format='png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    
    plot_confusion_matrix_pdr(df, classes,cm,accuracy, normalize=False,title=subtitle)
    plt.savefig(title2, format='png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    return accuracy
    
#Random Forest Classification
def my_random_forest(df1,src,dst,size_of_test,kind,path):#Kind can be "pdr", "mean_rssi" or "all"
    df=my_split(df1,src,dst,size_of_test,kind)
    subtitle="Accuracy = "
    if kind=="all":
        title=path+"RF\\RF_CM_ALL_"+str(src)+"==="+str(dst)+".png"
        title2=path+"RF\\RF_CM_Not_Normalized_ALL_"+str(src)+"==="+str(dst)+".png"
        colonne=['combined_features']
    elif kind=="pdr":
        title=path+"RF\\RF_CM_PDR_"+str(src)+"==="+str(dst)+".png"
        title2=path+"RF\\RF_CM_Not_Normalized_PDR_"+str(src)+"==="+str(dst)+".png"
        colonne=['pdr']
    else:
        title=path+"RF\\RF_CM_RSSI_"+str(src)+"==="+str(dst)+".png"
        title2=path+"RF\\RF_CM_Not_Normalized_RSSI_"+str(src)+"==="+str(dst)+".png"
        colonne=['mean_rssi']

    general_test_labels=list()
    general_pred_labels=list()
    channel_list=df1['channel'].unique()
    classes=df['target_names'].unique()
    accuracy_results=list()
    for i in range (len(channel_list)):
        channel_i=channel_list[i]
        colonne2=['target_names']
        ts=df.loc[(df['channel']==channel_i),colonne]
        ts2=df.loc[(df['channel']==channel_i),colonne2]

        features=ts.values
        labels=ts2['target_names']
        #classes=labels.unique()
        #if len(classes)>1:
        #    features, labels=rus.fit_resample(features, labels)#The ressampling strategy
        #    labels=pd.Series(labels)
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = size_of_test, random_state=None,shuffle=True)
        taill=len(pd.Series(train_labels).unique())
        if taill>1:
            rf = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=0)
            rf.fit(train_features, train_labels)
            pred_labels=rf.predict(test_features)
            general_test_labels=general_test_labels+list(test_labels)
            general_pred_labels=general_pred_labels+list(pred_labels)
        else:
            general_test_labels=general_test_labels+list(test_labels)
            general_pred_labels=general_pred_labels+list(test_labels)
    cm=confusion_matrix(general_test_labels,general_pred_labels)
    som=0
    taill=len(cm)
    total=0
    for i in range(taill):
        som=som+cm[i][i]
        for j in range(taill):
            total=total+cm[i][j]
        
    accuracy=som/total
    plot_confusion_matrix_pdr(df, classes,cm,accuracy, normalize=True,title=subtitle)
    
    plt.savefig(title, format='png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    
    plot_confusion_matrix_pdr(df, classes,cm,accuracy, normalize=False,title=subtitle)
    plt.savefig(title2, format='png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    return accuracy
    

def executor(df,src, dst, size_of_test,model_list,path):
    import os
    file_name=path+"final_results.csv"
    exists=os.path.isfile(file_name)
    if exists:
        file=open(file_name,"a+")
        #file.write(str('\n')+'model,link,test_size,accuracy_with_pdr_and_rssi,accuracy_with_pdr,accuracy_with_rssi')
    else:
        file=open(file_name,"a")
        file.write(str('\n')+'model,link,test_size,accuracy_with_pdr_and_rssi,accuracy_with_pdr,accuracy_with_rssi')
    for i in model_list:
        if i=='lr':
            lg_all=my_logreg(df,src,dst,size_of_test,'all',path)
            lg_pdr=my_logreg(df,src,dst,size_of_test,'pdr',path)
            lg_rssi=my_logreg(df,src,dst,size_of_test,'rssi',path)
            file.write(str('\n')+'LogReg,'+str(src)+'==='+str(dst)+','+str(size_of_test)+','+str(lg_all)+','+str(lg_pdr)+','+str(lg_rssi))
        elif i=='lsvm':
            lsvm_all=my_linear_svm(df,src,dst,size_of_test,'all',path)
            lsvm_pdr=my_linear_svm(df,src,dst,size_of_test,'pdr',path)
            lsvm_rssi=my_linear_svm(df,src,dst,size_of_test,'rssi',path)
            file.write(str('\n')+'LSVM,'+str(src)+'==='+str(dst)+','+str(size_of_test)+','+str(lsvm_all)+','+str(lsvm_pdr)+','+str(lsvm_rssi))
        elif i=='svm':
            svm_all=my_svm(df,src,dst,size_of_test,'all',path)
            svm_pdr=my_svm(df,src,dst,size_of_test,'pdr',path)
            svm_rssi=my_svm(df,src,dst,size_of_test,'rssi',path)
            file.write(str('\n')+'SVM,'+str(src)+'==='+str(dst)+','+str(size_of_test)+','+str(svm_all)+','+str(svm_pdr)+','+str(svm_rssi))
        elif i=='rf':  
            rf_all=my_random_forest(df,src,dst,size_of_test,'all',path)
            rf_pdr=my_random_forest(df,src,dst,size_of_test,'pdr',path)
            rf_rssi=my_random_forest(df,src,dst,size_of_test,'rssi',path)
            file.write(str('\n')+'RF,'+str(src)+'==='+str(dst)+','+str(size_of_test)+','+str(rf_all)+','+str(rf_pdr)+','+str(rf_rssi))
    file.close()
        
        
#The following function aims to compute the accuracies of all the links in the network for all our methods
def final_executor(df,size_of_test,my_execution_list,path):
    senders=df['src'].unique()
    receivers=df['dst'].unique()
    
    for sender in senders:
        for receiver in receivers:
            colonne=["pdr"]
            ts=df.loc[(df['src']==sender)&(df['dst']==receiver),colonne]
            if len(ts)>0:
                print("Starting link ",sender,"===",receiver)
                executor(df,sender,receiver,size_of_test,my_execution_list,path)
                print("Ending link ",sender,"===",receiver)






def my_general_predictor(df,size_of_test,kind,path):#Kind can be "pdr", "rssi" or "all"
    subtitle="Accuracy = "
    if kind=="all":
        lr_title="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\LR_CM_ALL.png"
        lr_title2="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\LR_CM_ALL_Not_Normalized.png"
        
        rf_title="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\RF_CM_ALL.png"
        rf_title2="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\RF_CM_ALL_Not_Normalized.png"
        
        svm_title="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\SVM_CM_ALL.png"
        svm_title2="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\SVM_ALL_Not_Normalized.png"
        
        lsvm_title="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\LSVM_CM_ALL.png"
        lsvm_title2="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\LSVM_CM_ALL_Not_Normalized.png"
        
        colonne=['combined_features']
        
    elif kind=="pdr":
        lr_title="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\LR_CM_PDR.png"
        lr_title2="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\LR_CM_PDR_Not_Normalized.png"
        
        rf_title="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\RF_CM_PDR.png"
        rf_title2="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\RF_CM_PDR_Not_Normalized.png"
        
        svm_title="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\SVM_CM_PDR.png"
        svm_title2="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\SVM_CM_PDR_Not_Normalized.png"
        
        lsvm_title="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\LSVM_CM_PDR.png"
        lsvm_title2="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\LSVM_CM_PDR_Not_Normalized.png"
        colonne=['pdr']
        
    else:
        lr_title="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\LR_CM_RSSI.png"
        lr_title2="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\LR_CM_RSSI_Not_Normalized.png"
        
        rf_title="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\RF_CM_RSSI.png"
        rf_title2="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\RF_CM_RSSI_Not_Normalized.png"
        
        svm_title="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\SVM_CM_RSSI.png"
        svm_title2="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\SVM_CM_RSSI_Not_Normalized.png"
        
        lsvm_title="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\LSVM_CM_RSSI.png"
        lsvm_title2="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\LSVM_CM_RSSI_Not_Normalized.png"
        colonne=['mean_rssi']

        
    lr_general_test_labels=list()
    lr_general_pred_labels=list()  
    
    rf_general_test_labels=list()
    rf_general_pred_labels=list()
    
    svm_general_test_labels=list()
    svm_general_pred_labels=list()
    
    lsvm_general_test_labels=list()
    lsvm_general_pred_labels=list()
    
    senders=df['src'].unique()
    receivers=df['dst'].unique()
    
    for sender in senders:
        for receiver in receivers:
            ts=my_split(df,sender,receiver,size_of_test,kind)
            if len(ts)>0:
                print('stating link: ',sender,'==>',receiver)
                channel_list=ts['channel'].unique()
                classes=ts['target_names'].unique()
                for i in range (len(channel_list)):
                    channel_i=channel_list[i]
                    colonne2=['target_names']
                    ts1=ts.loc[(ts['channel']==channel_i),colonne]
                    ts2=ts.loc[(ts['channel']==channel_i),colonne2]

                    features=ts1.values
                    labels=ts2['target_names']
                    #classes=labels.unique()
                    #if len(classes)>1:
                     #   features, labels=rus.fit_resample(features, labels)#The ressampling strategy
                     #   labels=pd.Series(labels)
                    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = size_of_test, random_state=None,shuffle=False)
                    taill=len(pd.Series(train_labels).unique())
                    if taill>1:
                        LogReg=LogisticRegression()
                        LogReg.fit(train_features,train_labels)
                        lr_pred_labels=LogReg.predict(test_features)
                        lr_general_test_labels=lr_general_test_labels+list(test_labels)
                        lr_general_pred_labels=lr_general_pred_labels+list(lr_pred_labels)
                        
                        lsvm = LinearSVC(random_state=0, tol=1e-5)
                        lsvm.fit(train_features, train_labels)
                        lsvm_pred_labels = lsvm.predict(test_features)
                        lsvm_general_test_labels=lsvm_general_test_labels+list(test_labels)
                        lsvm_general_pred_labels=lsvm_general_pred_labels+list(lsvm_pred_labels)
                        
                        svm = SVC(random_state=0, tol=1e-5)
                        svm.fit(train_features, train_labels)
                        svm_pred_labels = svm.predict(test_features)
                        svm_general_test_labels=svm_general_test_labels+list(test_labels)
                        svm_general_pred_labels=svm_general_pred_labels+list(svm_pred_labels)
                        
                        rf = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=0)
                        rf.fit(train_features, train_labels)
                        rf_pred_labels = rf.predict(test_features)
                        rf_general_test_labels=rf_general_test_labels+list(test_labels)
                        rf_general_pred_labels=rf_general_pred_labels+list(rf_pred_labels)
                        
                    else:
                        lr_general_test_labels=lr_general_test_labels+list(test_labels)
                        lr_general_pred_labels=lr_general_pred_labels+list(test_labels)
                        
                        rf_general_test_labels=rf_general_test_labels+list(test_labels)
                        rf_general_pred_labels=rf_general_pred_labels+list(test_labels)
                        
                        svm_general_test_labels=svm_general_test_labels+list(test_labels)
                        svm_general_pred_labels=svm_general_pred_labels+list(test_labels)
                        
                        lsvm_general_test_labels=lsvm_general_test_labels+list(test_labels)
                        lsvm_general_pred_labels=lsvm_general_pred_labels+list(test_labels)
                print('ending link: ',sender,'==>',receiver)
    lr_cm=confusion_matrix(lr_general_test_labels,lr_general_pred_labels)
    
    svm_cm=confusion_matrix(svm_general_test_labels,svm_general_pred_labels)
    
    lsvm_cm=confusion_matrix(lsvm_general_test_labels,lsvm_general_pred_labels)
    
    rf_cm=confusion_matrix(rf_general_test_labels,rf_general_pred_labels)
    
    lr_som=0
    lsvm_som=0
    rf_som=0
    svm_som=0
    taill=len(lr_cm)
    total=0
    for i in range(taill):
        lr_som=lr_som+lr_cm[i][i]
        svm_som=svm_som+svm_cm[i][i]
        lsvm_som=lsvm_som+lsvm_cm[i][i]
        rf_som=rf_som+rf_cm[i][i]
        
        for j in range(taill):
            total=total+lr_cm[i][j]
        
    lr_accuracy=lr_som/total
    lsvm_accuracy=lsvm_som/total
    svm_accuracy=svm_som/total
    rf_accuracy=rf_som/total
    
    plot_confusion_matrix_pdr(df, classes,lr_cm,lr_accuracy, normalize=True,title=subtitle)
    plot_confusion_matrix_pdr(df, classes,rf_cm,rf_accuracy, normalize=True,title=subtitle)
    plot_confusion_matrix_pdr(df, classes,lsvm_cm,lsvm_accuracy, normalize=True,title=subtitle)
    plot_confusion_matrix_pdr(df, classes,svm_cm,svm_accuracy, normalize=True,title=subtitle)
    
    plt.savefig(lr_title, format='png', bbox_inches='tight', pad_inches=0)
    plt.savefig(lsvm_title, format='png', bbox_inches='tight', pad_inches=0)
    plt.savefig(svm_title, format='png', bbox_inches='tight', pad_inches=0)
    plt.savefig(rf_title, format='png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    
    plot_confusion_matrix_pdr(df, classes,lr_cm,lr_accuracy, normalize=False,title=subtitle)
    plot_confusion_matrix_pdr(df, classes,lsvm_cm,lsvm_accuracy, normalize=False,title=subtitle)
    plot_confusion_matrix_pdr(df, classes,svm_cm,svm_accuracy, normalize=False,title=subtitle)
    plot_confusion_matrix_pdr(df, classes,rf_cm,rf_accuracy, normalize=False,title=subtitle)
    
    plt.savefig(lr_title2, format='png', bbox_inches='tight', pad_inches=0)
    plt.savefig(lsvm_title2, format='png', bbox_inches='tight', pad_inches=0)
    plt.savefig(svm_title2, format='png', bbox_inches='tight', pad_inches=0)
    plt.savefig(rf_title2, format='png', bbox_inches='tight', pad_inches=0)
    
    plt.clf()
    return lr_accuracy,lsvm_accuracy,rf_accuracy,svm_accuracy

myLoc = RLock() #A lock to synchronize writing operations inside test and pred labels
class MyThread(Thread):
    def __init__(self,ts,general_test_labels,general_pred_labels,colonne,size_of_test,model):
        Thread.__init__(self)
        self.ts=ts
        self.general_test_labels=general_test_labels
        self.general_pred_labels=general_pred_labels
        self.colonne=colonne
        self.size_of_test=size_of_test
        self.model=model
        
    def run(self):
        if len(self.ts)>0:
            channel_list=self.ts['channel'].unique()
            classes=self.ts['target_names'].unique()
            for i in range (len(channel_list)):
                channel_i=channel_list[i]
                colonne2=['target_names']
                ts1=self.ts.loc[(self.ts['channel']==channel_i),self.colonne]
                ts2=self.ts.loc[(self.ts['channel']==channel_i),colonne2]

                features=ts1.values
                labels=ts2['target_names']
                #classes=labels.unique()
                #if len(classes)>1:
                 #   features, labels=rus.fit_resample(features, labels)#The ressampling strategy
                 #   labels=pd.Series(labels)
                train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = self.size_of_test, random_state=None,shuffle=True)
                taill=len(pd.Series(train_labels).unique())
                if taill>1:
                    if self.model='lsvm':
                        lsvm = LinearSVC(random_state=0, tol=1e-5)
                        lsvm.fit(train_features, train_labels)
                        pred_labels=lsvm.predict(test_features)
                    elif self.model='rf':
                        rf = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=0)
                        rf.fit(train_features, train_labels)
                        pred_labels=rf.predict(test_features)
                    elif self.model='svm':
                        clf = SVC(random_state=0, tol=1e-5)
                        clf.fit(train_features, train_labels)
                        pred_labels=clf.predict(test_features)
                    else:
                        LogReg=LogisticRegression()
                        LogReg.fit(train_features,train_labels)
                        pred_labels=LogReg.predict(test_features)
                    with myLoc:
                        self.general_test_labels=self.general_test_labels+list(test_labels)
                        self.general_pred_labels=self.general_pred_labels+list(pred_labels)
                else:
                    with myLoc:
                        self.general_test_labels=self.general_test_labels+list(test_labels)
                        self.general_pred_labels=self.general_pred_labels+list(test_labels)
                    
                    
                    
def my_particular_split(df,src,dst,size_of_test,kind):
    ros = RandomOverSampler(random_state=0)
    rus = RandomUnderSampler(random_state=0)
    target=list()
    target_names=list()
    
    if kind=="all":                                              #it is for pdr+rssi
        combined_features=list()
        for i in range(len(df)):
            combined_features.append(df['pdr'][i]*df['pdr'][i]+df['mean_rssi'][i]*df['mean_rssi'][i])
            if df['pdr'][i] <= 0.3:
                target.append(0)
                target_names.append('Bad')
            elif df['pdr'][i]>=0.75:
                target.append(2)
                target_names.append('Good')
            elif ((df['pdr'][i]>0.3 and df['pdr'][i]<0.75)and df['mean_rssi'][i]<=-87):
                target.append(0)
                target_names.append('Bad')
            else:
                target.append(1)
                target_names.append('Interm.')

        df['combined_features']=combined_features
        colonne=['combined_features']
    elif kind=="pdr":                                            #It is for pdr
        for i in range(len(df)):
            if df['pdr'][i] <= 0.3 :
                target.append(0)
                target_names.append('Bad')
            elif df['pdr'][i]>=0.75 :
                target.append(2)
                target_names.append('Good')
            else:
                target.append(1)
                target_names.append('Interm.')
            colonne=['pdr']
    else:                                                          #It is for mean_rssi
        for i in range(len(df)):
            if df['mean_rssi'][i] <= -87 :
                target.append(0)
                target_names.append('Bad')
            elif df['mean_rssi'][i]>=-85 :
                target.append(2)
                target_names.append('Good')
            else:
                target.append(1)
                target_names.append('Interm.')
        colonne=['mean_rssi']
    df['target']=target
    df['target_names']=target_names
    colonne2=['target_names']
    df=df.loc[(df['src']==src)&(df['dst']==dst)]

    return df


def my_general_linear_svm_with_threads(df,size_of_test,kind,path):#Kind can be "pdr", "rssi" or "all"
    text=""
    subtitle="Accuracy = "
    if kind=="all":
        title="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\LSVM_CM_ALL.png"
        title2="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\LSVM_CM_ALL_Not_Normalized.png"
        colonne=['combined_features']
        
    elif kind=="pdr":
        title="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\LSVM_CM_PDR.png"
        title2="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\LSVM_CM_PDR_Not_Normalized.png"
        colonne=['pdr']
        
    else:
        title="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\LSVM_CM_RSSI.png"
        title2="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\LSVM_CM_RSSI_Not_Normalized.png"
        colonne=['mean_rssi']

        
    general_test_labels=list()
    general_pred_labels=list()  
    senders=df['src'].unique()
    receivers=df['dst'].unique()
    
    for sender in senders:
        for receiver in receivers:
            print('starting link: ',sender,'==>',receiver)
            ts=my_particular_split(df,sender,receiver,size_of_test,kind)
            mySolver=MyThread(ts,general_test_labels,general_pred_labels,colonne,size_of_test,'lsvm')# For each link, we create a new thread to proccess it
            mySolver.start()
            mySolver.join()
            print('ending link: ',sender,'==>',receiver)
            
    cm=confusion_matrix(general_test_labels,general_pred_labels)
    som=0
    taill=len(cm)
    total=len(general_test_labels)
    for i in range(taill):
        som=som+cm[i][i]

            
        
    accuracy=som/total
    plot_confusion_matrix_pdr(df, classes,cm,accuracy, normalize=True,title=subtitle)
    
    plt.savefig(title, format='png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    
    plot_confusion_matrix_pdr(df, classes,cm,accuracy, normalize=False,title=subtitle)
    plt.savefig(title2, format='png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    return accuracy


def my_general_random_forest_with_threads(df,size_of_test,kind,path):#Kind can be "pdr", "rssi" or "all"
    text=""
    subtitle="Accuracy = "
    if kind=="all":
        title="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\RF_CM_ALL.png"
        title2="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\RF_CM_ALL_Not_Normalized.png"
        colonne=['combined_features']
        
    elif kind=="pdr":
        title="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\RF_CM_PDR.png"
        title2="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\RF_CM_PDR_Not_Normalized.png"
        colonne=['pdr']
        
    else:
        title="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\RF_CM_RSSI.png"
        title2="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\RF_CM_RSSI_Not_Normalized.png"
        colonne=['mean_rssi']

        
    general_test_labels=list()
    general_pred_labels=list()  
    senders=df['src'].unique()
    receivers=df['dst'].unique()
    
    for sender in senders:
        for receiver in receivers:
            print('starting link: ',sender,'==>',receiver)
            ts=my_particular_split(df,sender,receiver,size_of_test,kind)
            mySolver=MyThread(ts,general_test_labels,general_pred_labels,colonne,size_of_test,'rf')# For each link, we create a new thread to proccess it
            mySolver.start()
            mySolver.join()
            print('ending link: ',sender,'==>',receiver)
            
    cm=confusion_matrix(general_test_labels,general_pred_labels)
    som=0
    taill=len(cm)
    total=len(general_test_labels)
    for i in range(taill):
        som=som+cm[i][i]

            
        
    accuracy=som/total
    plot_confusion_matrix_pdr(df, classes,cm,accuracy, normalize=True,title=subtitle)
    
    plt.savefig(title, format='png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    
    plot_confusion_matrix_pdr(df, classes,cm,accuracy, normalize=False,title=subtitle)
    plt.savefig(title2, format='png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    return accuracy




#ros = RandomOverSampler(random_state=0)
#rus = RandomUnderSampler(random_state=0)
path="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\GENERAL\\"
path2="C:\\Users\WSN-LINK\\Documents\\TEST\\SOCALE\\CHOOSE\\FINAL\\"
my_execution_list=['lr', 'lsvm','svm','rf']
my_execution_list1=['lr']

#print("Accuracy of PDR+RSSI [lr,lsvm,rf,svm] = ",my_general_predictor(df,0.25,"all",path))
#print("Accuracy of PDR [lr,lsvm,rf,svm] = ",my_general_predictor(df,0.25,"pdr",path))
#print("Accuracy of RSSI [lr,lsvm,rf,svm] = ",my_general_predictor(df,0.25,"rssi",path))

my_general_linear_svm_with_threads(df,0.25,'pdr',path)
#executor(df,19, 44, 0.25,my_execution_list,path2)
#executor(df,3, 15, 0.25,my_execution_list,path)
#final_executor(df,0.25,my_execution_list,path)


# In[ ]:





# In[ ]:




