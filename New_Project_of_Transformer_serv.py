#!/usr/bin/env python
# coding: utf-8

# In[2]:


'''from google.colab import drive
drive.mount('/content/drive')'''


# In[3]:


'''!ls drive/MyDrive/'''


# In[68]:


import pandas as pd
df_concat = pd.read_csv("Complete_Data (1).csv")
df_concat.head()


# In[69]:


df_concat.shape


# In[ ]:





# In[70]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
tf.version


# In[71]:


#df_concat=train_in[0:100000]
df_concat = df_concat.set_index('Time')


# In[72]:


# how long preceding sequence to consider for prediction
SEQ_LEN = 10           

# How far into the future we are making prediction(10 min in this case)
FUTURE_PERIOD = 1

RATIO_TO_PREDICT = "Close"
def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


# In[73]:


times = sorted(df_concat.index.values)  # get the times
last_10 = sorted(df_concat.index.values)[-int(0.1*len(times))] 
last_20 = sorted(df_concat.index.values)[-int(0.2*len(times))]

test_df = df_concat[-50000:] 
validation_df = df_concat[-100000:-50000] 
train_df = df_concat[:-100000] 
from collections import deque
import numpy as np
import random
     

#dfX_train = train_df.drop(columns=["target"])
#dfX_test = test_df.drop(columns=["target"])
#dfX_valid = validation_df.drop(columns=["target"])

train_ = train_df.values
valid_ = validation_df.values
test_ = test_df.values
     

print("train shape {0}".format(train_.shape))
print("valid shape {0}".format(valid_.shape))
print("test shape {0}".format(test_.shape))


# In[74]:


scaler = MinMaxScaler()
scale_close = MinMaxScaler()
     

x = train_[:,3].copy()
scale_close.fit(x.reshape(-1, 1))
scaler.fit(train_)

train_ = scaler.transform(train_)

valid_ = scaler.transform(valid_)
test_ = scaler.transform(test_)


# In[75]:


scaler = MinMaxScaler()
scale_close = MinMaxScaler()
     

x = train_[:,3].copy()
scale_close.fit(x.reshape(-1, 1))
     
MinMaxScaler(copy=True, feature_range=(0, 1))

scaler.fit(train_)

train_ = scaler.transform(train_)

valid_ = scaler.transform(valid_)
test_ = scaler.transform(test_)


# In[76]:


def split_data(data):
    X = []
    Y = []
    for i in range(SEQ_LEN, len(data)-FUTURE_PERIOD+1):
        X.append(data[i-SEQ_LEN:i])
        Y.append(data[i+(FUTURE_PERIOD-1), 3])
    return np.array(X), np.array(Y)


# In[77]:


#del train_in 


# In[78]:


X_train, y_train = split_data(train_)
X_test, y_test = split_data(test_)
X_valid, y_valid = split_data(valid_)
     

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 70))
X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 70))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 70))
     

y_train.shape


# In[79]:


print("train shape {0}".format(X_train.shape))
print("valid shape {0}".format(X_valid.shape))
print("test shape {0}".format(X_test.shape))


# In[80]:


X_train_2, y_train_2 = split_data(train_)
X_train_2 = np.reshape(X_train_2, (X_train_2.shape[0], X_train_2.shape[1], 70))
     
## show predictions
plt.figure(figsize=(15, 5))

plt.plot(np.arange(y_train_2.shape[0]), y_train_2, color='blue', label='train target')

plt.plot(np.arange(y_train_2.shape[0], y_train_2.shape[0]+y_valid.shape[0]), y_valid,
         color='gray', label='valid target')

plt.plot(np.arange(y_train_2.shape[0]+y_valid.shape[0],
                   y_train_2.shape[0]+y_valid.shape[0]+y_test.shape[0]),
         y_test, color='black', label='test target')


plt.title('Train set')
plt.xlabel('time [minutes]')
plt.ylabel('normalized price')
plt.legend(loc='best')
     


# In[81]:


import random, os, sys
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.layers import Layer
     
class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)
    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    def compute_output_shape(self, input_shape):
        return input_shape

class ScaledDotProductAttention():
    def __init__(self, d_model, attn_dropout=0.1):
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)
    def __call__(self, q, k, v, mask):
        attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/self.temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x:(-1e+10)*(1-x))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn

class MultiHeadAttention():
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0, use_norm=True):
        self.mode = mode
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        if mode == 0:
            self.qs_layer = Dense(n_head*d_k, use_bias=False)
            self.ks_layer = Dense(n_head*d_k, use_bias=False)
            self.vs_layer = Dense(n_head*d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization() if use_norm else None
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, d_k])
                x = tf.transpose(x, [2, 0, 1, 3])  
                x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
                return x
            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x:K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)  
                
            def reshape2(x):
                s = tf.shape(x)   # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]]) 
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1], n_head*d_v])  # [batch_size, len_v, n_head * d_v]
                return x
            head = Lambda(reshape2)(head)
        elif self.mode == 1:
            heads = []; attns = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)   
                ks = self.ks_layers[i](k) 
                vs = self.vs_layers[i](v) 
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head); attns.append(attn)
            head = Concatenate()(heads) if n_head > 1 else heads[0]
            attn = Concatenate()(attns) if n_head > 1 else attns[0]

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        if not self.layer_norm: return outputs, attn
        # outputs = Add()([outputs, q]) # sl: fix
        return self.layer_norm(outputs), attn

class PositionwiseFeedForward():
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)
    def __call__(self, x):
        output = self.w_1(x) 
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)

class EncoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
    def __call__(self, enc_input, mask=None):
        output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
        output = self.pos_ffn_layer(output)
        return output, slf_attn

def GetPosEncodingMatrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] 
        if pos != 0 else np.zeros(d_emb) 
            for pos in range(max_len)
            ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
    return pos_enc

def GetPadMask(q, k):
    ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
    mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
    mask = K.batch_dot(ones, mask, axes=[2,1])
    return mask

def GetSubMask(s):
    len_s = tf.shape(s)[1]
    bs = tf.shape(s)[:1]
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask


# In[85]:


class CustomeLearningSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomeLearningSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        param_1 = tf.math.rsqrt(step)
        param_2 = step * (self.warmup_steps**(-1.5))
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(param_1, param_2)
     

sample_learning_rate = CustomeLearningSchedule(d_model=128)

plt.plot(sample_learning_rate(tf.range(200000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")


# In[87]:


D_MODEL=300

lr = CustomeLearningSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)
     

def build_model():
    inp = Input(shape = (SEQ_LEN, 70))

    x = Bidirectional(LSTM(128, return_sequences=True))(inp)
    x = Bidirectional(LSTM(64, return_sequences=True))(x) 
        
    #for i in range(2):
    x, self_attn = EncoderLayer(
            d_model=D_MODEL,
            d_inner_hid=512, 
            n_head=4, 
            d_k=64,
            d_v=64, 
            dropout=0.2)(x)
        
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(128, activation="relu")(conc)
    x = Dense(1, activation="sigmoid")(conc)      

    model = Model(inputs = inp, outputs = x)
    model.compile(
        loss = "mean_squared_error",  metrics=['acc'],
        optimizer = optimizer)
    
    return model
     

multi_head = build_model()
multi_head.summary()


# In[88]:


callback = EarlyStopping(monitor='val_loss',
                         patience=3,
                         restore_best_weights=True)
history = multi_head.fit(x=X_train, 
                         y=y_train,
                         batch_size=2000,
                         epochs=200,
                         validation_data=(X_valid, y_valid), 
                         callbacks=[callback])


# In[111]:



predicted_stock_price_multi_head = multi_head.predict(X_train)

predicted_stock_price_multi_head.shape


# In[116]:


predicted_stock_price = np.vstack((np.full((1,1), np.nan), predicted_stock_price_multi_head))
'''plt.plot(y_test, color = 'black', label = ' Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Price')
plt.title('Close Price Prediction', fontsize=30)
plt.xlabel('DateTime')
plt.ylabel('Close Price') 
plt.legend(fontsize=18)
plt.show()'''


# In[117]:


plt.plot(y_test[-10000:], color = 'black', label = ' Stock Price')


# In[92]:


df_concat.head()


# In[120]:


len(predicted_stock_price),X_train.shape,len(df_concat)


# In[123]:


import pandas as pd

col_names =  ['Close', 'Predicted', 'Long','Short']

my_df  = pd.DataFrame(columns = col_names)

my_df['Close']=df_concat['Close'][:2533108]
my_df['Predicted']=predicted_stock_price[1:]
my_df['Date']=my_df.index

my_df.tail()


# In[124]:



maxr_df=my_df


# In[125]:



from scipy.signal import argrelextrema
n = 30
maxr_df['min_10'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.less_equal,
                    order=n)[0]]['Close']
maxr_df['max_10'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.greater_equal,
                    order=n)[0]]['Close']
from scipy.signal import argrelextrema
n = 50
maxr_df['min_20'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.less_equal,
                    order=n)[0]]['Close']
maxr_df['max_20'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.greater_equal,
                    order=n)[0]]['Close']
n = 100
maxr_df['min_30'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.less_equal,
                    order=n)[0]]['Close']
maxr_df['max_30'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.greater_equal,
                    order=n)[0]]['Close']
n = 150
maxr_df['min_40'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.less_equal,
                    order=n)[0]]['Close']
maxr_df['max_40'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.greater_equal,
                    order=n)[0]]['Close']
n = 500
maxr_df['min_50'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.less_equal,
                    order=n)[0]]['Close']
maxr_df['max_50'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.greater_equal,
                    order=n)[0]]['Close']
n = 600
maxr_df['min_60'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.less_equal,
                    order=n)[0]]['Close']
maxr_df['max_60'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.greater_equal,
                    order=n)[0]]['Close']
n = 900
maxr_df['min_70'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.less_equal,
                    order=n)[0]]['Close']
maxr_df['max_70'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.greater_equal,
                    order=n)[0]]['Close']
n = 1000
maxr_df['min_80'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.less_equal,
                    order=n)[0]]['Close']
maxr_df['max_80'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.greater_equal,
                    order=n)[0]]['Close']
n = 1200
maxr_df['min_90'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.less_equal,
                    order=n)[0]]['Close']
maxr_df['max_90'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.greater_equal,
                    order=n)[0]]['Close']


# In[127]:


'''# Find local peaks / values with noise reduction
n = 1000
maxr_df['min_500'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.less_equal,
                    order=n)[0]]['Close']
maxr_df['max_500'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.greater_equal,
                    order=n)[0]]['Close']

# Plot results
plt.scatter(maxr_df.Date, maxr_df['min_500'], c='r')
plt.scatter(maxr_df.Date, maxr_df['max_500'], c='g')
maxr_df.Predicted.plot()'''


# In[138]:



import operator
def Long(l):
  min_index, min_value = min(enumerate(l), key=operator.itemgetter(1))
  return min_index+1
def Short(l):
    import numpy as np
    if l[-1]==l[0] and l[-1]!=0:
        ind =l.count(l[-1])
    elif l[1]!=0 and l.count(l[1])>1:
        ind =l.count(l[1])
    else:
        ind, min_value = max(enumerate(l), key=operator.itemgetter(1))
    return ind+1


# In[139]:



maxr_df=maxr_df.fillna(0)


# In[ ]:



Long_l=[]
Short_l=[]
for index, row in maxr_df.iterrows():
  Long_l.append(Long([row['min_10'],row['min_20'],row['min_30'],row['min_40'],row['min_50'],row['min_60'],row['min_70'],row['min_80'],row['min_90']]))
  Short_l.append(Short([row['max_10'],row['max_20'],row['max_30'],row['max_40'],row['max_50'],row['max_60'],row['max_70'],row['max_80'],row['max_90']]))
maxr_df['Short']=Short_l
maxr_df['Long']=Long_l


# In[133]:


maxr_df=maxr_df.drop([ 'min_10', 'max_10',
       'min_20', 'max_20', 'min_30', 'max_30', 'min_40', 'max_40', 'min_50',
       'max_50', 'min_60', 'max_60', 'min_70', 'max_70', 'min_80', 'max_80',
       'min_90', 'max_90','Predicted','Date'],axis=1)
maxr_df.head()


# In[134]:


set(maxr_df['Short'])


# In[135]:



maxr_df.to_csv('Predict_Trining_.csv',sep='\t')


# In[143]:


multi_head.summary()


# In[144]:


from tensorflow.keras.models import Model
#train_X, Y_train, epochs=100, batch_size=100, validation_data=(test_X, Y_test), verbose=1, shuffle=False)
model = multi_head # include here your original model X_train, X_test, Y_train, Y_test

layer_name = 'dense_4'
model_feat = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)

feat_extract = model_feat.predict(X_train)
print(feat_extract.shape)


# In[147]:


Extrac_data=pd.DataFrame(feat_extract)

Extrac_data['Price']=maxr_df['Close'].values
Extrac_data['Long']=maxr_df['Long'].values
Extrac_data['Short']=maxr_df['Short'].values
Extrac_data.to_csv("Features.csv",sep='\t')
Extrac_data.tail()


# In[148]:


import requests
import pandas as pd
import json
from datetime import datetime, timezone, timedelta
import time
import os

URL = 'http://172.16.10.10:8080/webhook'
PARAMS = { 'date' : pd.to_datetime(int((datetime.now(timezone.utc).replace(microsecond=0, second=0) - timedelta(days=30)).timestamp())*1000, unit='ms')
         ,'username': 'ashkan'
         , 'password' : 'mashayekhi'
         }
tic = time.time()
data = requests.get(url = URL, params=PARAMS)
tac = time.time()
print(tac-tic)
data = pd.DataFrame(data.json())
data['Time'] = pd.to_datetime(data['Time'])
print(data)
print(type(data))
data.to_csv('daily3.csv')


# In[181]:


print(data.tail(100))


# In[150]:


data=data.drop('tag',axis=1)
data.head()


# In[152]:


data.set_index('Time', inplace=True)


# In[156]:


train_=data.values


# In[157]:


scaler = MinMaxScaler()
scale_close = MinMaxScaler()
x = train_[:,3].copy()
scale_close.fit(x.reshape(-1, 1))
     
MinMaxScaler(copy=True, feature_range=(0, 1))

scaler.fit(train_)

train_ = scaler.transform(train_)


# In[160]:


X_train, y_train = split_data(train_)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 70))


# In[162]:


predicted_stock_price_multi_head = multi_head.predict(X_train)

predicted_stock_price_multi_head.shape


# In[167]:


import pandas as pd
predicted_stock_price = np.vstack((np.full((1,1), np.nan), predicted_stock_price_multi_head))
len(data),predicted_stock_price.shape


# In[182]:



col_names =  ['Close', 'Predicted', 'Long','Short']

my_df  = pd.DataFrame(columns = col_names)

my_df['Close']=data['Close'][9:]
my_df['Predicted']=predicted_stock_price
my_df['Date']=my_df.index

my_df.tail()


# In[177]:



maxr_df=my_df
from scipy.signal import argrelextrema
n = 30
maxr_df['min_10'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.less_equal,
                    order=n)[0]]['Close']
maxr_df['max_10'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.greater_equal,
                    order=n)[0]]['Close']
from scipy.signal import argrelextrema
n = 50
maxr_df['min_20'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.less_equal,
                    order=n)[0]]['Close']
maxr_df['max_20'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.greater_equal,
                    order=n)[0]]['Close']
n = 100
maxr_df['min_30'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.less_equal,
                    order=n)[0]]['Close']
maxr_df['max_30'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.greater_equal,
                    order=n)[0]]['Close']
n = 150
maxr_df['min_40'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.less_equal,
                    order=n)[0]]['Close']
maxr_df['max_40'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.greater_equal,
                    order=n)[0]]['Close']
n = 500
maxr_df['min_50'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.less_equal,
                    order=n)[0]]['Close']
maxr_df['max_50'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.greater_equal,
                    order=n)[0]]['Close']
n = 600
maxr_df['min_60'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.less_equal,
                    order=n)[0]]['Close']
maxr_df['max_60'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.greater_equal,
                    order=n)[0]]['Close']
n = 900
maxr_df['min_70'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.less_equal,
                    order=n)[0]]['Close']
maxr_df['max_70'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.greater_equal,
                    order=n)[0]]['Close']
n = 1000
maxr_df['min_80'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.less_equal,
                    order=n)[0]]['Close']
maxr_df['max_80'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.greater_equal,
                    order=n)[0]]['Close']
n = 1440
maxr_df['min_90'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.less_equal,
                    order=n)[0]]['Close']
maxr_df['max_90'] = maxr_df.iloc[argrelextrema(maxr_df.Close.values, np.greater_equal,
                    order=n)[0]]['Close']
import operator
def Long(l):
  min_index, min_value = min(enumerate(l), key=operator.itemgetter(1))
  return min_index+1
def Short(l):
    import numpy as np
    if l[-1]==l[0] and l[-1]!=0:
        ind =l.count(l[-1])
    elif l[1]!=0 and l.count(l[1])>1:
        ind =l.count(l[1])
    else:
        ind, min_value = max(enumerate(l), key=operator.itemgetter(1))
    return ind+1
maxr_df=maxr_df.fillna(0)
Long_l=[]
Short_l=[]
for index, row in maxr_df.iterrows():
  Long_l.append(Long([row['min_10'],row['min_20'],row['min_30'],row['min_40'],row['min_50'],row['min_60'],row['min_70'],row['min_80'],row['min_90']]))
  Short_l.append(Short([row['max_10'],row['max_20'],row['max_30'],row['max_40'],row['max_50'],row['max_60'],row['max_70'],row['max_80'],row['max_90']]))
maxr_df['Short']=Short_l
maxr_df['Long']=Long_l


# In[178]:


set(maxr_df['Short'])
set(maxr_df['Long'])


# In[179]:


maxr_df=maxr_df.drop([ 'min_10', 'max_10',
       'min_20', 'max_20', 'min_30', 'max_30', 'min_40', 'max_40', 'min_50',
       'max_50', 'min_60', 'max_60', 'min_70', 'max_70', 'min_80', 'max_80',
       'min_90', 'max_90','Predicted','Date'],axis=1)
maxr_df.head()
maxr_df.to_csv('Predict_3Day_.csv',sep='\t')


# In[ ]:




