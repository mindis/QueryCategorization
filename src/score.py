import pandas as pd
import numpy as np
import scipy
import pickle
from sparse_dot_topn import awesome_cossim_topn

from sklearn.feature_extraction.text import TfidfVectorizer
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

def score_knn(args):
    
    df_test = pd.read_csv(args.input, names=['query'])
    df_train = pd.read_csv('./data/trainSet.csv', names=['query','category'])
    df_test['query_lem']=df_test['query'].apply(lemmatize)
    df_train['query_lem']=df_train['query'].apply(lemmatize)  
    
    vectorizer=TfidfVectorizer()
    train_features=vectorizer.fit_transform(df_train.query_lem.values)
    test_features=vectorizer.transform(df_test.query_lem.values)
    matches = awesome_cossim_topn(test_features, train_features.transpose(), 20, 0.01)
    ind=np.argwhere(matches)
    i1=ind[:,0]
    i2=ind[:,1]

    df_test.reset_index(inplace=True)
    df_test=df_test.rename(columns={'index':'index1'})

    index1=np.take(df_test['index1'].values, i1)
    query=np.take(df_test['query'].values, i1)
    categories=np.take(df_train.category.values, i2)

    df2=pd.DataFrame(data={'query':query, 'index1':index1, 'cat':categories})

#     most frequent category
    pred=df2[['query','index1','cat']].groupby(['query','index1']).agg(lambda x:scipy.stats.mode(x)[0])
    pred=pred.reset_index()
    pred=pred.sort_values('index1')
    pred[['query','cat']].to_csv('./data/pred_knn.csv', index=None, header=False)
    logger.info("Finished scoring test sample with KNN")
    
    
    
def score_lr(args):
    df_test = pd.read_csv(args.input, names=['query'])
    df_test['query_lem']=df_test['query'].apply(lemmatize)
    df_test['vw_test']='| ' + df_test['query_lem'].values
    test_examples=list(df_test['vw_test'].values)   
    
    logger.info("Calculating accuracy and F1 score on hold out data set")    
    vw = pyvw.vw("-i ./models/lr.vw -t")
    pred = [vw.predict(sample) for sample in test_examples]
    
#     shifting back by one
    df_test['pred']=pred
    df_test['pred']=df_test['pred']-1
    df_test[['query','pred']].to_csv('./data/pred_lr.csv', index=None, header=False)
    logger.info("Finished scoring test sample with LR")
    
def run_scoring(args):
        
    if args.model=='knn':
        logger.info("Scoring KNN model")
        score_knn(args)

    if args.model=='lr':
        score_lr(args)

