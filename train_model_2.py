import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# normalaize value of features
# save that scalers to use it later
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


def train_test_split(df_concat, test_num):
    # how long preceding sequence to consider for prediction
    SEQ_LEN = 1

    test_df = df_concat[-test_num:]
    train_df = df_concat[:-test_num]

    train_ = train_df.values
    test_ = test_df.values

    X_train, y_train = split_data(train_)
    X_test, y_test = split_data(test_)

    y_train = y_train.reshape((-1, SEQ_LEN, 1))
    y_test = y_test.reshape((-1, SEQ_LEN, 1))

    return X_train, y_train, X_test, y_test


def split_data(data):
    SEQ_LEN = 1
    FUTURE_PERIOD = 15
    X = []
    Y = []
    for i in range(SEQ_LEN, len(data) - FUTURE_PERIOD + 1):
        X.append(data[i - SEQ_LEN:i])
        Y.append(round(data[i + (FUTURE_PERIOD - 1), 3],4))
    return np.array(X), np.array(Y)


@tf.keras.utils.register_keras_serializable()
class PositionEncoding(tf.keras.layers.Layer):
    def __init__(self, dim, *args, **kwargs):
        super(PositionEncoding, self).__init__(*args, **kwargs)
        self.dim = dim
        self.inv_freq = np.float32(1 / np.power(10000, np.arange(0, self.dim, 2) // dim))

    def get_config(self):
        config = {
            "dim": self.dim
        }
        base_config = super(PositionEncoding, self).get_config()
        config = {**base_config, **config}
        return config

    def call(self, x):
        max_len = tf.shape(x)[1]
        pos_x = tf.range(max_len, dtype=x.dtype)
        pos_enc = tf.einsum("i,j->ij", pos_x, self.inv_freq)
        pos_enc = tf.stack((tf.sin(pos_enc), tf.cos(pos_enc)), -1)
        pos_enc = tf.reshape(pos_enc, (*pos_enc.shape[:-2], -1))
        return x + pos_enc


@tf.keras.utils.register_keras_serializable()
class transformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, ffn_dim, projection_dim, dropout_rate, *args, **kwargs):
        super(transformerBlock, self).__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.projection_dim)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.drop1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.drop2 = tf.keras.layers.Dropout(self.dropout_rate)
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(self.ffn_dim, activation=tf.keras.layers.PReLU()),
                                        tf.keras.layers.Dense(self.projection_dim)])

    def get_config(self):
        config = super(transformerBlock, self).get_config()
        config.update({
            "num_heads": self.num_heads, "ffn_dim": self.ffn_dim, "projection_dim": self.projection_dim,
            "dropout_rate": self.dropout_rate
        })
        return config

    def call(self, inputs, training):
        attn_output = self.attn(inputs, inputs, training=training)
        attn_output = self.drop1(attn_output, training=training)
        out1 = self.ln1(inputs + attn_output, training=training)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.drop2(ffn_output, training=training)
        return self.ln2(out1 + ffn_output, training=training)


def get_model():
    SEQ_LEN = 1
    inp = tf.keras.Input((SEQ_LEN, 70))
    x = PositionEncoding(70)(inp)
    # x = tf.keras.layers.Dense(64, activation="relu")(x)
    for _ in range(15):
        x = transformerBlock(num_heads=4, ffn_dim=128, projection_dim=70, dropout_rate=0.2)(x)

    # avg = tf.keras.layers.GlobalAvgPool1D()(x)
    # max = tf.keras.layers.GlobalMaxPool1D()(x)
    # x = tf.keras.layers.Concatenate(axis=-1)([max, avg])

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation="relu"))(x)

    out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation="sigmoid"))(x)

    # x = tf.keras.layers.Dense(128, activation="relu")(x)
    # out = tf.keras.layers.Dense(15, activation="tanh")(x)
    return tf.keras.Model(inp, out)


if __name__ == '__main__':
    # read DataBase that must be trained
    df_concat = pd.read_csv("./module/data/Complete_Data.csv")
    df_concat = df_concat.set_index('Time')
    df_concat.dropna(inplace=True)
    # apply MinMaxScaler on 66 features
    df_concat = normalize_data(df_concat, need_update_scalers=True)
    # split data to Train and Test
    X_train, y_train, X_test, y_test = train_test_split(df_concat, test_num=10000)

    # build and compile model
    model = get_model()
    model.compile(loss="mean_squared_error", metrics=["accuracy"], optimizer="adam")
    # early stop if 3 epoch not change
    es = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=5, restore_best_weights=True)
    # fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=10000, epochs=4, callbacks=[es])
    # save model
    model.save("./module/models/transformer_model_15Min2_100epo.tf")
