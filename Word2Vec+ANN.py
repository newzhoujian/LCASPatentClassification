import pandas as pd
import tensorflow as tf
import sys
import numpy as np
import keras
import keras.backend as K
from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import h5py
from keras.models import save_model
import sys
import pickle
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


f = open('../../data/alldata80000_new.pkl', 'rb')
X, y, word2vec_metrix = pickle.load(f)
f.close()
X_train, X_val, X_test, y_train, y_val, y_test = X[:50000], X[50000:60000], X[60000:], y[:50000], y[50000:60000], y[60000:]

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.per_process_gpu_memory_fraction = 1.
sess = tf.Session(config=config)
KTF.set_session(sess)

sent_maxlen = 200
word_size = 300
sent_size = 300
sess_size = 200
batch_size = 200


with tf.device('/gpu:0'):
    
    cnt = 60000
    model_input = Input(shape=(sent_maxlen,))
    
    wordembed = Embedding(len(word2vec_metrix), 300, weights=[word2vec_metrix], input_length=200, trainable=False)(model_input)
    wordembed = Flatten()(wordembed)
    wordembed = Dense(600, activation='relu')(wordembed)
    conv_out = Dropout(0.5)(wordembed)
    model_output = Dense(8, activation='softmax')(conv_out)
    model = Model(inputs=model_input, outputs=model_output)
    model.summary()
    checkpoint = ModelCheckpoint('../../PCmodel/Word2VecANN.h5', monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    history = model.fit(X_train, y_train, batch_size=200, epochs=20, validation_data=[X_val, y_val], callbacks=[checkpoint])
    model.save('../../PCmodel/Word2VecANN.h5')
    y_pred = model.predict(X_test)

y1 = [np.argmax(i) for i in y_test]
y2 = [np.argmax(i) for i in y_pred]

from sklearn import metrics

print('macro_precision:\t',end='')
print(metrics.precision_score(y1,y2,average='macro'))

print('macro_recall:\t\t',end='')
print(metrics.recall_score(y1,y2,average='macro'))

print('macro_f1:\t\t',end='')
print(metrics.f1_score(y1,y2,average='macro'))