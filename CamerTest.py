#!/usr/bin/env python
# coding: utf-8

# In[16]:


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
#import MyAnalysis 
rcParams['figure.figsize'] = 4,3

#Import for threading

import random
import sys
from threading import Thread, RLock
import time

import numpy as np
df = pd.read_table("grenoble_71.k7",sep = ',',header = 0)

#Methods that help to draw the confusion matrix

def trace_conf_mat(cm, acc,first,classes, norm, title,cmap=plt.cm.Blues):
    accuracy=acc
    if norm:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    chaine='Predicted label\n'
    for i in range(len(first)):
        chaine=chaine+'\n'+first[i]
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title+'%.2f' % accuracy,
           ylabel='True label',
           xlabel=chaine)
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

def plot_confusion_matrix_pdr(df, classes, cm,accuracy,first, normalize=False, title=None,cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    trace_conf_mat(cm, accuracy,first,classes, normalize, title,cmap=plt.cm.Blues)
np.set_printoptions(precision=2)

#Split method, used to prepare classes for classification tasks

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

#Function that computes the Accuracy, the precision, the recall and the F1-Score given a confusion matrix

def computer(cm,classes):
    taill=len(cm)
    

    #We compute total for precision and recall
    total_precision=list()#Total for precision
    total_recall=list()#Total for recall
    total=0#Total for accuracy
    som=0#For accuracy
    for j in range(taill):
        total1=0
        total2=0
        som=som+cm[j][j]
        for k in range(taill):
            total1=total1+cm[j][k]
            total2=total2+cm[k][j]
            total=total+cm[j][k]
            #print('Totally après: ',totali)
            #print('cm[',j,'][',k,']: ',cm[j][k])
        #print('Totally après: ',total1)
        #print('Totally 2 après: ',total2)
        total_precision.append(total1)
        total_recall.append(total2)
    accuracy=som/total
        
    #Value of precision and recall returned for each classes
    returner=list()
    for i in range(taill):
        precision=cm[i][i]/total_precision[i]
        recall=cm[i][i]/total_recall[i]
        f1_Score=2*(precision*recall)/(precision+recall)
        vale='['+classes[i]+']: precision=%.2f' % precision+', recall=%.2f' %recall+', f1-score=%.2f'%f1_Score
        returner.append(vale)
    return accuracy,returner


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
        #Classification over each channel of the link
        channel_i=channel_list[i]
        colonne2=['target_names']
        ts=df.loc[(df['channel']==channel_i),colonne]
        ts2=df.loc[(df['channel']==channel_i),colonne2]

        features=ts.values
        labels=ts2['target_names']
        classes2=labels.unique()
        if len(classes2)>1:
        #   features, labels=rus.fit_resample(features, labels)#The ressampling strategy
         #   labels=pd.Series(labels)

         #The train_test_split() method works only if the number of classes for prediction is greather or aqual to 2
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
        else:
            #If the number of class for classification is equal to one in a given channel of a link, we just report the corresponding value of classes in test_labels and pred_label, according to the size_test
            test_tail=int(size_of_test*len(labels))
            test_labels=list()
            pred_labels=list()
            labels=labels.to_numpy()
            for j in range(test_tail):
                test_labels.append(labels[j])
                pred_labels.append(labels[j])
            general_test_labels=general_test_labels+test_labels
            general_pred_labels=general_pred_labels+pred_labels

    cm=confusion_matrix(general_test_labels,general_pred_labels)
    accuracy,returner=computer(cm,classes)
    
    #print("Returner:  ",returner)
    plot_confusion_matrix_pdr(df, classes,cm,accuracy,returner, normalize=True,title=subtitle)
    
    plt.savefig(title, format='png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    
    plot_confusion_matrix_pdr(df, classes,cm,accuracy,returner, normalize=False,title=subtitle)
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
        classes2=labels.unique()
        if len(classes2)>1:
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
        else:
        #If the number of class for classification is equal to one in a given channel of a link, we just report the corresponding value of classes in test_labels and pred_label, according to the size_test
            test_tail=int(size_of_test*len(labels))
            test_labels=list()
            pred_labels=list()
            labels=labels.to_numpy()
            for j in range(test_tail):
                test_labels.append(labels[j])
                pred_labels.append(labels[j])
            general_test_labels=general_test_labels+test_labels
            general_pred_labels=general_pred_labels+pred_labels
    cm=confusion_matrix(general_test_labels,general_pred_labels)
    accuracy,returner=computer(cm,classes)
    
    #print("Returner:  ",returner)
    plot_confusion_matrix_pdr(df, classes,cm,accuracy,returner, normalize=True,title=subtitle)
    
    plt.savefig(title, format='png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    
    plot_confusion_matrix_pdr(df, classes,cm,accuracy,returner, normalize=False,title=subtitle)
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
        classes2=labels.unique()
        if len(classes2)>1:
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
        else:
        #If the number of class for classification is equal to one in a given channel of a link, we just report the corresponding value of classes in test_labels and pred_label, according to the size_test
            test_tail=int(size_of_test*len(labels))
            test_labels=list()
            pred_labels=list()
            labels=labels.to_numpy()
            for j in range(test_tail):
                test_labels.append(labels[j])
                pred_labels.append(labels[j])
            general_test_labels=general_test_labels+test_labels
            general_pred_labels=general_pred_labels+pred_labels
    cm=confusion_matrix(general_test_labels,general_pred_labels)
    accuracy,returner=computer(cm,classes)
    
    #print("Returner:  ",returner)
    plot_confusion_matrix_pdr(df, classes,cm,accuracy,returner, normalize=True,title=subtitle)
    
    plt.savefig(title, format='png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    
    plot_confusion_matrix_pdr(df, classes,cm,accuracy,returner, normalize=False,title=subtitle)
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
        classes2=labels.unique()
        if len(classes2)>1:
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
        else:
        #If the number of class for classification is equal to one in a given channel of a link, we just report the corresponding value of classes in test_labels and pred_label, according to the size_test
            test_tail=int(size_of_test*len(labels))
            test_labels=list()
            pred_labels=list()
            labels=labels.to_numpy()
            for j in range(test_tail):
                test_labels.append(labels[j])
                pred_labels.append(labels[j])
            general_test_labels=general_test_labels+test_labels
            general_pred_labels=general_pred_labels+pred_labels
    cm=confusion_matrix(general_test_labels,general_pred_labels)
    accuracy,returner=computer(cm,classes)
    
    #print("Returner:  ",returner)
    plot_confusion_matrix_pdr(df, classes,cm,accuracy,returner, normalize=True,title=subtitle)
    
    plt.savefig(title, format='png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    
    plot_confusion_matrix_pdr(df, classes,cm,accuracy,returner, normalize=False,title=subtitle)
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
        lr_title=path+"\\LR_CM_ALL.png"
        lr_title2=path+"\\LR_CM_ALL_Not_Normalized.png"
        
        rf_title=path+"\\RF_CM_ALL.png"
        rf_title2=path+"\\RF_CM_ALL_Not_Normalized.png"
        
        svm_title=path+"\\SVM_CM_ALL.png"
        svm_title2=path+"\\SVM_ALL_Not_Normalized.png"
        
        lsvm_title=path+"\\LSVM_CM_ALL.png"
        lsvm_title2=path+"\\LSVM_CM_ALL_Not_Normalized.png"
        
        colonne=['combined_features']
        
    elif kind=="pdr":
        lr_title=path+"\\GENERAL\\LR_CM_PDR.png"
        lr_title2=path+"\\LR_CM_PDR_Not_Normalized.png"
        
        rf_title=path+"\\RF_CM_PDR.png"
        rf_title2=path+"\\RF_CM_PDR_Not_Normalized.png"
        
        svm_title=path+"\\SVM_CM_PDR.png"
        svm_title2=path+"\\SVM_CM_PDR_Not_Normalized.png"
        
        lsvm_title=path+"\\LSVM_CM_PDR.png"
        lsvm_title2=path+"\\LSVM_CM_PDR_Not_Normalized.png"
        colonne=['pdr']
        
    else:
        lr_title=path+"\\LR_CM_RSSI.png"
        lr_title2=path+"\\LR_CM_RSSI_Not_Normalized.png"
        
        rf_title=path+"\\RF_CM_RSSI.png"
        rf_title2=path+"\\RF_CM_RSSI_Not_Normalized.png"
        
        svm_title=path+"\\SVM_CM_RSSI.png"
        svm_title2=path+"\\SVM_CM_RSSI_Not_Normalized.png"
        
        lsvm_title=path+"\\LSVM_CM_RSSI.png"
        lsvm_title2=path+"\\LSVM_CM_RSSI_Not_Normalized.png"
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
                    classes2=labels.unique()
                    if len(classes2)>1:
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
                    else:
                    #If the number of class for classification is equal to one in a given channel of a link, we just report the corresponding value of classes in test_labels and pred_label, according to the size_test
                        test_tail=int(size_of_test*len(labels))
                        test_labels=list()
                        pred_labels=list()
                        labels=labels.to_numpy()
                        for j in range(test_tail):
                            test_labels.append(labels[j])
                            pred_labels.append(labels[j])

                        lr_general_test_labels=lr_general_test_labels+test_labels
                        lr_general_pred_labels=lr_general_pred_labels+pred_labels
                        
                        rf_general_test_labels=rf_general_test_labels+test_labels
                        rf_general_pred_labels=rf_general_pred_labels+pred_labels
                        
                        svm_general_test_labels=svm_general_test_labels+test_labels
                        svm_general_pred_labels=svm_general_pred_labels+pred_labels
                        
                        lsvm_general_test_labels=lsvm_general_test_labels+test_labels
                        lsvm_general_pred_labels=lsvm_general_pred_labels+pred_labels
                print('ending link: ',sender,'==>',receiver)
    lr_cm=confusion_matrix(lr_general_test_labels,lr_general_pred_labels)
    
    svm_cm=confusion_matrix(svm_general_test_labels,svm_general_pred_labels)
    
    lsvm_cm=confusion_matrix(lsvm_general_test_labels,lsvm_general_pred_labels)
    
    rf_cm=confusion_matrix(rf_general_test_labels,rf_general_pred_labels)
    
    lr_accuracy,lr_returner=computer(lr_cm,classes)
    lsvm_accuracy,lsvm_returner=computer(lsvm_cm,classes)
    svm_accuracy,svm_returner=computer(svm_cm,classes)
    rf_accuracy,rf_returner=computer(rf_cm,classes)

    plot_confusion_matrix_pdr(df, classes,cm,accuracy,returner, normalize=True,title=subtitle)

    plot_confusion_matrix_pdr(df, classes,lr_cm,lr_accuracy,lr_returner, normalize=True,title=subtitle)
    plot_confusion_matrix_pdr(df, classes,rf_cm,rf_accuracy,rf_accuracy, normalize=True,title=subtitle)
    plot_confusion_matrix_pdr(df, classes,lsvm_cm,lsvm_accuracy,lsvm_returner, normalize=True,title=subtitle)
    plot_confusion_matrix_pdr(df, classes,svm_cm,svm_accuracy,svm_accuracy, normalize=True,title=subtitle)
    
    plt.savefig(lr_title, format='png', bbox_inches='tight', pad_inches=0)
    plt.savefig(lsvm_title, format='png', bbox_inches='tight', pad_inches=0)
    plt.savefig(svm_title, format='png', bbox_inches='tight', pad_inches=0)
    plt.savefig(rf_title, format='png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    
    plot_confusion_matrix_pdr(df, classes,lr_cm,lr_accuracy,lr_accuracy, normalize=False,title=subtitle)
    plot_confusion_matrix_pdr(df, classes,lsvm_cm,lsvm_accuracy,lsvm_accuracy, normalize=False,title=subtitle)
    plot_confusion_matrix_pdr(df, classes,svm_cm,svm_accuracy,svm_accuracy, normalize=False,title=subtitle)
    plot_confusion_matrix_pdr(df, classes,rf_cm,rf_accuracy,rf_accuracy, normalize=False,title=subtitle)
    
    plt.savefig(lr_title2, format='png', bbox_inches='tight', pad_inches=0)
    plt.savefig(lsvm_title2, format='png', bbox_inches='tight', pad_inches=0)
    plt.savefig(svm_title2, format='png', bbox_inches='tight', pad_inches=0)
    plt.savefig(rf_title2, format='png', bbox_inches='tight', pad_inches=0)
    
    plt.clf()
    return lr_accuracy,lsvm_accuracy,rf_accuracy,svm_accuracy


path="C:\\Users\Hp\\Documents\\TEST\\FINAL\\GENERAL\\"
my_execution_list=['lr', 'lsvm','svm','rf']
my_execution_list1=['lr']

#my_random_forest(df,0, 20, 0.25,"all",path)
my_general_predictor(df,0.25,all,path)


# In[ ]:




