import os

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import sys

sys.setrecursionlimit(10000)

df_concat = pd.read_csv("./module/data/new_data.csv")
df_concat = df_concat.set_index('Time')
df_concat.dropna(inplace=True)

features = df_concat.columns[4:-15]
scaler = MinMaxScaler()
for f in features:
    scaler.fit(df_concat[[f]])
    filename = "./module/models/scalers/scaler_" + f + ".pickle"
    with open(filename, 'wb') as handle:
        pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
    df_concat[f] = scaler.transform(df_concat[[f]])
# how long preceding sequence to consider for prediction
SEQ_LEN = 1
# How far into the future we are making prediction(10 min in this case)
FUTURE_PERIOD = 1

RATIO_TO_PREDICT = "Close"

times = sorted(df_concat.index.values)  # get the times
last_10 = sorted(df_concat.index.values)[-int(0.1 * len(times))]
last_20 = sorted(df_concat.index.values)[-int(0.2 * len(times))]

test_df = df_concat[-5000:]
validation_df = df_concat[-10000:-5000]
train_df = df_concat[:-10000]

train_ = train_df.values
valid_ = validation_df.values
test_ = test_df.values


def split_data(data):
    X = []
    Y = []
    for i in range(SEQ_LEN, len(data) - FUTURE_PERIOD + 1):
        X.append(data[i - SEQ_LEN:i, :-15])
        Y.append(data[i + (FUTURE_PERIOD - 1), -15:])
    return np.array(X), np.array(Y)


X_train, y_train = split_data(train_)
X_test, y_test = split_data(test_)
X_valid, y_valid = split_data(valid_)


class LayerNormalization(Layer):
    def __init__(self, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)

    # def get_config(self):
    #
    #     config = super().get_config().copy()
    #     config.update({
    #         'eps': self.eps,
    #     })
    #     return config
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + 1e-6) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class ScaledDotProductAttention():
    def __init__(self, d_model, attn_dropout=0.1, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)

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
        self.attention.__name__ = 'ScaledDotProductAttention'

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


@tf.keras.utils.register_keras_serializable()
class PositionwiseFeedForward():
    def __init__(self, d_hid, d_inner_hid, dropout=0.1,*args, **kwargs):
        super(PositionwiseFeedForward, self).__init__(*args, **kwargs)
        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def get_config(self):
        config = super(PositionwiseFeedForward, self).get_config()
        return {"w_1": self.w_1, "w_2": self.w_2, "layer_norm": self.layer_norm, "dropout": self.dropout, **config}

    def __call__(self, x):
        output = self.w_1(x)
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)


@tf.keras.utils.register_keras_serializable()
class EncoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1,*args, **kwargs):
        super(EncoderLayer, self).__init__(*args, **kwargs)

        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn_layer = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        return {"self_att_layer": self.self_att_layer, "pos_ffn_layer": self.pos_ffn_layer, **config}

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


@tf.keras.utils.register_keras_serializable()
class CustomeLearningSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomeLearningSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def get_config(self):
        config = super(CustomeLearningSchedule, self).get_config()
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps, **config}
    def __call__(self, step):
        param_1 = tf.math.rsqrt(step)
        param_2 = step * (self.warmup_steps ** (-1.5))
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(param_1, param_2)


def build_model():

    inp = Input(shape=(SEQ_LEN, 70))
    x = Dense(128, activation="relu")(inp)
    # x = Bidirectional(LSTM(128, return_sequences=True))(inp)
    # x = Bidirectional(LSTM(64, return_sequences=True))(x)

    for i in range(6):
        x, self_attn = EncoderLayer(
            d_model=D_MODEL,
            d_inner_hid=512,
            n_head=15,
            d_k=64,
            d_v=64,
            dropout=0.2)(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(128, activation="relu")(conc)
    x = Dense(15, activation="tanh")(conc)

    model = Model(inputs=inp, outputs=x)
    model.compile(
        loss="mean_squared_error", metrics=['acc'],
        optimizer=optimizer)

    return model



sample_learning_rate = CustomeLearningSchedule(d_model=128)
D_MODEL = 300

lr = CustomeLearningSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)
multi_head = build_model()
callback = EarlyStopping(monitor='val_loss',
                         patience=3,
                         restore_best_weights=True)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="./module/models/transformer_model/checkpoints",
                                                 save_weights_only=True,
                                                 verbose=1)
multi_head.fit(x=X_train,
               y=y_train,
               batch_size=10000,
               epochs=50,
               validation_data=(X_valid, y_valid),
               callbacks=[callback])


# if __name__ == '__main__':


# , custom_objects={'ScaledDotProductAttention': ScaledDotProductAttention}
# tf.keras.models.save_model(multi_head, "./module/models/transformer_model.tf")
multi_head.save("./module/models/transformer_model2.h5")
# with open("./module/models/transformer_model.pkl", 'wb') as handle:
#     pickle.dump(multi_head, handle)
# pickle.dump(multi_head, open('model.pkl', 'wb'))
# joblib.dump(multi_head, 'model.pkl')
# module_no_signatures_path = os.path.join("./module/models/", 'transformer_model')
# print('Saving model...')
# tf.saved_model.save(multi_head, module_no_signatures_path)
