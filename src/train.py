import pandas as pd
import numpy as np
import yaml
import scipy
import pickle

from sparse_dot_topn import awesome_cossim_topn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

import nltk
from nltk.stem import WordNetLemmatizer
from vowpalwabbit import pyvw

from src.process import lemmatize


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)


def train_knn(args):
    
    df = pd.read_csv(args.input, names=['query','category'])

    logger.info("Lemmatizing and preparing data for KNN model")
    df['query_lem']=df['query'].apply(lemmatize)
    df_train, df_test=train_test_split(df, test_size=0.2, stratify=df.category.values, random_state=42)
        
    logger.info("Fitting vectorizer")
    vectorizer=TfidfVectorizer()
    train_features=vectorizer.fit_transform(df_train.query_lem.values)
    logger.info("Scoring 20% holdout sample") 
    test_features=vectorizer.transform(df_test.query_lem.values)
    matches = awesome_cossim_topn(test_features, train_features.transpose(), 20, 0.01)
    ind=np.argwhere(matches)
    i1=ind[:,0]
    i2=ind[:,1]

    df_test.reset_index(inplace=True)
    df_test=df_test.rename(columns={'index':'index1'})

    index1=np.take(df_test['index1'].values, i1)
    categories=np.take(df_train.category.values, i2)

    
    df2=pd.DataFrame(data={'index1':index1, 'cat':categories})

#     most frequent category
    pred=df2[['index1','cat']].groupby(['index1']).agg(lambda x:scipy.stats.mode(x)[0])
    pred=pred.reset_index()
    pred=pred.sort_values('index1')
    
    pred1=pred.merge(df_test[['index1', 'category']].rename(columns={'category':'cat_true'}), on='index1', how='left')
    pred1['match']=(pred1.cat==pred1.cat_true).astype(int)
    logger.info("Hold out sample accuracy %s", np.round(pred1['match'].mean(),2))
    logger.info("Hold out sample f1 score %s", np.round(f1_score(pred1.cat_true.values, pred1.cat.values, average='weighted'), 2))
    logger.info("Hold out sample sklearn accuracy %s", np.round(accuracy_score(pred1.cat_true.values,pred1.cat.values), 2))

    
    
    
    
    
    
def train_lr(args):
    
    logger.info("Reading data")
    df = pd.read_csv(args.input, names=['query','category'])
    
    logger.info("Lemmatizing and preparing data")
    df['query_lem']=df['query'].apply(lemmatize)
    
#     ensure that category start from 1
    df.category=df.category+1
    df['vw_train']=df.category.astype(str)+ ' | ' + df['query_lem'].values
    df['vw_test']='| ' + df['query_lem'].values
    
    category_count=len(df.category.unique())
    
    df_train, df_test=train_test_split(df, test_size=0.2, stratify=df.category.values, random_state=42)
    
    train_examples=list(df_train['vw_train'].values)
    test_examples=list(df_test['vw_train'].values)    
    
    logger.info("Training LR model")
    vw_command="--oaa {} --random_seed 17 --cache_file ./tmp1 -b 27 -f ./models/lr.vw ".format(category_count)
    logger.info(vw_command)
    vw = pyvw.vw(vw_command)
    for iteration in range(2):
        logger.info("Iteration %s", iteration)
        for i in range(len(train_examples)):
            vw.learn(train_examples[i])
    vw.finish()    
    logger.info("Finished model training")
    
    logger.info("Calculating accuracy and F1 score on hold out data set")    
    vw = pyvw.vw("-i ./models/lr.vw  -t")
    pred = [vw.predict(sample) for sample in test_examples]
    logger.info("LR holdout accuracy score is %s", np.round(accuracy_score(df_test.category.values,pred),2))
    logger.info("LR holdout F1 score is %s", np.round(f1_score(df_test.category.values,pred, average='weighted'),2))   
    
    
def run_training(args):
    
    if args.model=='knn':
        logger.info("Training KNN model")
        train_knn(args)
        
    if args.model=='lr':
        logger.info("Training Logistic Regression model")
        train_lr(args)
        