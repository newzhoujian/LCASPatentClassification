# LCASPatentClassification

## 处理数据
这三个文件是处理数据的

ProcessData.ipynb 

ProcessData1.ipynb 

RandomData.ipynb


ProcessData.ipynb 和 ProcessData1.ipynb 得到word2id，id化后的专利数据和标签以及词向量等数据。

RandomData.ipynb 随机打乱id化后的专利数据


## 模型
### 深度学习模型（七个）
Word2Vec+ANN.py

Word2Vec+ATT.py

Word2Vec+BiGRU+ATT+TextCNN.py

Word2Vec+BiGRU+TextCNN.py

Word2Vec+BiGRU.py

Word2Vec+GRU.py

Word2Vec+TextCNN.py


### 传统机器学习模型（三个）
TFIDF+LR:DT:RF.py

## 专利数据（8万条）
patentData_80000_20180622.xlsx
