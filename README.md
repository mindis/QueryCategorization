# Solution of Adthena query categorization task

Solution implements two models: 
1) KNN with cosine similarity distance
2) Logistic Regression from VW  with hashing trick to reduce memory requirements

Scores for test sample are in ```./data ```

Usage instructions to train and score models:

0. Create virtual environment to install dependencies
```
virtualenv -p python3 query_cat
source query_cat/bin/activate
```
1. Clone repository, change to folder  and install dependencies
```
git clone https://github.com/mindis/QueryCategorization/
cd QueryCategorization
pip3 install -r requirements.txt
```
2. Within folder run following command to download data:
```
python3 run.py load
```

3. Train and evaluate KNN and LR models
```
python3 run.py train -i ./data/trainSet.csv -m knn
python3 run.py train -i ./data/trainSet.csv -m lr
````
5. Score KNN and LR models
```
python3 run.py score -i ./data/candidateTestSet.csv -m knn
python3 run.py score -i ./data/candidateTestSet.csv -m lr
```

Answers
1. Due to large number of categories, short queries and noisy labels I've tried simple top20 cosine KNN and logistic regression models. 
2. From out of box preprocessing tools only lemmatization improved accuracy significantly. 
3. F1 and accuracy
4. top20 KNN is fast and only requires to optimize number of neighbours. LR with Vowpall Wabbit allows to balance memory, speed and accuracy. sklearn LR run out of memory.
5. Weaknesses: 

    KNN similarity measure is not optimized for predictive accuracy. Similarity based on trained embeddings potentially could work better(e.g. Prototypical and Matching networks)

    VW LR options were not optimized, can be improved with trainable embeddings, ngrams, interactions. 

6. Better data cleaning, preprocessing and hand crafted features could also help:

    Translate  queries to english

    Create better quality labels.

    Put more emphasis on nouns and named entities (keywords),

7. Just testing
