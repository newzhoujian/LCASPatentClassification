import pandas as pd
import sys
import numpy as np
import pickle
import os
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics



df_file = pd.read_excel('../../data/patentData_80000_20180622.xlsx')
df_file['X'] = df_file['标题'] + '。' + df_file['摘要'] + '。' + df_file['首项权利要求']

import jieba
stopwordfile = open('../../dict/StopWords_CON.txt', 'r', encoding='utf-8')
def getstopword():
    w = set()
    for line in stopwordfile:
        line.strip().split('\n')
        w.add(line[:len(line)-1].strip())
    return w

stopwordset = getstopword()

def cutWords(sentence):
    word_list = jieba.cut(sentence)
    res = ' '.join(word_list)
    res = res.split(' ')
    tempX = ''
    for i in res:
        if i not in stopwordset:
            tempX+=i
            tempX+=' '
    return tempX.strip()


df_file['X'] = df_file['X'].apply(cutWords)

label2id = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7
}

def transy(x):
    return label2id[x[0]]

df_file['y'] = df_file['主IPC分类号-小类'].apply(transy)
y = df_file['y'].values

document = df_file['X'].values

tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit_transform(document)
X_train, X_test, y_train, y_test = train_test_split(tfidf_model, y, test_size=0.25, random_state=42)

models = {}
models['LR'] = LogisticRegression()
models['LR'].fit(X_train[:], y_train[:])

models['DT'] = DecisionTreeClassifier()
models['DT'].fit(X_train[:], y_train[:])

models['RF'] = RandomForestClassifier()
models['RF'].fit(X_train[:], y_train[:])

for key in models.keys():
	y_pred = models[key].predict(X_test)
	print(key)
	
	print('macro_precision:\t',end='')
	print(metrics.precision_score(y_test, y_pred,average='macro'))

	print('macro_recall:\t\t',end='')
	print(metrics.recall_score(y_test, y_pred,average='macro'))

	print('macro_f1:\t\t',end='')
	print(metrics.f1_score(y_test, y_pred,average='macro'))

