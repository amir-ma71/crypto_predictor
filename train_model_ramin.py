import operator
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import os
import tensorflow as tf
from keras.initializers.initializers_v2 import Ones, Zeros
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
import tensorflow as tf
import keras.backend as K
from tensorflow.python.keras.layers import Layer


def normalize_data(df_concat, need_update_scalers=False):
    features = df_concat.columns
    scaler = MinMaxScaler()
    # if need to rebuild scalers and update it
    if need_update_scalers:
        for f in features:
            scaler.fit(df_concat[[f]])
            filename = "./module/models/scalers/scaler_" + f + ".pickle"
            with open(filename, 'wb') as handle:
                pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
            df_concat[f] = scaler.transform(df_concat[[f]])
    # just use saved scalers
    else:
        for f in features:
            filename = "./module/models/scalers/scaler_" + f + ".pickle"
            with open(filename, 'rb') as handle:
                scaler = pickle.load(handle)
            df_concat[f] = scaler.transform(df_concat[[f]])

    return df_concat


df_concat = pd.read_csv("./module/data/Complete_Data.csv")
df_concat = df_concat.set_index('Time')
df_concat.dropna(inplace=True)

df_concat = normalize_data(df_concat, need_update_scalers=True)

times = sorted(df_concat.index.values)  # get the times
last_10 = sorted(df_concat.index.values)[-int(0.1 * len(times))]
last_20 = sorted(df_concat.index.values)[-int(0.2 * len(times))]

test_df = df_concat[-500:]
validation_df = df_concat[-700:-500]
train_df = df_concat[-2000:-700]

train_ = train_df.values
valid_ = validation_df.values
test_ = test_df.values
# how long preceding sequence to consider for prediction
SEQ_LEN = 1

# How far into the future we are making prediction(10 min in this case)
FUTURE_PERIOD = 15

RATIO_TO_PREDICT = "Close"


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


def split_data(data):
    X = []
    Y = []
    for i in range(SEQ_LEN, len(data) - FUTURE_PERIOD + 1):
        X.append(data[i - SEQ_LEN:i])
        Y.append(round(data[i + (FUTURE_PERIOD - 1), 3],4))
    return np.array(X), np.array(Y)


X_train, y_train = split_data(train_)
X_test, y_test = split_data(test_)
X_valid, y_valid = split_data(valid_)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 70))
X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 70))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 70))

X_train_2, y_train_2 = split_data(train_)
X_train_2 = np.reshape(X_train_2, (X_train_2.shape[0], X_train_2.shape[1], 70))


## show predictions
# plt.figure(figsize=(15, 5))
#
# plt.plot(np.arange(y_train_2.shape[0]), y_train_2, color='blue', label='train target')
#
# plt.plot(np.arange(y_train_2.shape[0], y_train_2.shape[0] + y_valid.shape[0]), y_valid,
#          color='gray', label='valid target')
#
# plt.plot(np.arange(y_train_2.shape[0] + y_valid.shape[0],
#                    y_train_2.shape[0] + y_valid.shape[0] + y_test.shape[0]),
#          y_test, color='black', label='test target')
#
# plt.title('Train set')
# plt.xlabel('time [minutes]')
# plt.ylabel('normalized price')
# plt.legend(loc='best')
#
# plt.show()


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
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]) / self.temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x: (-1e+10) * (1 - x))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
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
            self.qs_layer = Dense(n_head * d_k, use_bias=False)
            self.ks_layer = Dense(n_head * d_k, use_bias=False)
            self.vs_layer = Dense(n_head * d_v, use_bias=False)
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
                s = tf.shape(x)  # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, d_k])
                x = tf.transpose(x, [2, 0, 1, 3])
                x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
                return x

            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x: K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)

            def reshape2(x):
                s = tf.shape(x)  # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]])
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1], n_head * d_v])  # [batch_size, len_v, n_head * d_v]
                return x

            head = Lambda(reshape2)(head)
        elif self.mode == 1:
            heads = [];
            attns = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)
                ks = self.ks_layers[i](k)
                vs = self.vs_layers[i](v)
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head);
                attns.append(attn)
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
        self.pos_ffn_layer = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

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
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


def GetPadMask(q, k):
    ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
    mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
    mask = K.batch_dot(ones, mask, axes=[2, 1])
    return mask


def GetSubMask(s):
    len_s = tf.shape(s)[1]
    bs = tf.shape(s)[:1]
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask


class CustomeLearningSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomeLearningSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        param_1 = tf.math.rsqrt(step)
        param_2 = step * (self.warmup_steps ** (-1.5))
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(param_1, param_2)


sample_learning_rate = CustomeLearningSchedule(d_model=128)

D_MODEL = 300

lr = CustomeLearningSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)


def build_model():
    inp = Input(shape=(SEQ_LEN, 70))

    x = Bidirectional(LSTM(128, return_sequences=True))(inp)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)

    # for i in range(2):
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

    model = Model(inputs=inp, outputs=x)
    model.compile(
        loss="mean_squared_error", metrics=['acc'],
        optimizer=optimizer)

    return model


multi_head = build_model()

callback = EarlyStopping(monitor='val_loss',
                         patience=3,
                         restore_best_weights=True)
history = multi_head.fit(x=X_train,
                         y=y_train,
                         batch_size=2000,
                         epochs=1,
                         validation_data=(X_valid, y_valid),
                         callbacks=[callback])

predicted_stock_price_multi_head = multi_head.predict(X_test)
predicted_stock_price = np.vstack((np.full((1, 1), np.nan), predicted_stock_price_multi_head))

col_names = ['Close', 'Predicted', 'Long', 'Short']

my_df = pd.DataFrame(columns=col_names)

my_df['Close'] = df_concat['Close'][:len(X_test)]
my_df['Predicted'] = predicted_stock_price[1:]
my_df['Date'] = my_df.index



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

Long_l=[]
Short_l=[]
for index, row in maxr_df.iterrows():
  Long_l.append(Long([row['min_10'],row['min_20'],row['min_30'],row['min_40'],row['min_50'],row['min_60'],row['min_70'],row['min_80'],row['min_90']]))
  Short_l.append(Short([row['max_10'],row['max_20'],row['max_30'],row['max_40'],row['max_50'],row['max_60'],row['max_70'],row['max_80'],row['max_90']]))

maxr_df['Short']=Short_l
maxr_df['Long']=Long_l

maxr_df=maxr_df.drop([ 'min_10', 'max_10',
       'min_20', 'max_20', 'min_30', 'max_30', 'min_40', 'max_40', 'min_50',
       'max_50', 'min_60', 'max_60', 'min_70', 'max_70', 'min_80', 'max_80',
       'min_90', 'max_90','Predicted','Date'],axis=1)


set(maxr_df['Short'])


maxr_df.to_csv('Predict_Trining_.csv',sep='\t')

model = multi_head # include here your original model X_train, X_test, Y_train, Y_test

layer_name = 'dense_4'
model_feat = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)

feat_extract = model_feat.predict(X_test)
print(feat_extract.shape)


# In[147]:


Extrac_data=pd.DataFrame(feat_extract)

Extrac_data['Price']=maxr_df['Close'].values
Extrac_data['Long']=maxr_df['Long'].values
Extrac_data['Short']=maxr_df['Short'].values
Extrac_data.to_csv("Features.csv",sep='\t')