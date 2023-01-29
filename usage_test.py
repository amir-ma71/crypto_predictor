import pickle
import pandas as pd
import tensorflow as tf
from train_model_2 import get_model, train_test_split, normalize_data

# read DataBase
df_concat = pd.read_csv("./module/data/Complete_Data.csv")
df_concat = df_concat.set_index('Time')
# Normalize features
df_concat = normalize_data(df_concat, need_update_scalers=False)
# train and test split
X_train, y_train, X_test, y_test = train_test_split(df_concat, test_num=10000)



# load trained model
model = tf.keras.models.load_model('./module/models/transformer_model_15Min2.tf', custom_objects={'get_model': get_model})
# predict test data
y_pred = model.predict(X_test)

# build test Dataframe
out_dict = {}
pred_feature = [df_concat.columns[3]]



for j in range(len(y_test)):
    i = 0
    for fe in pred_feature:
        if fe not in out_dict.keys():
            out_dict[fe] = []
            out_dict[fe+"_pred"] = []
        out_dict[fe].append(y_test[j][0][i])
        out_dict[fe+"_pred"].append(y_pred[j][0][i])
        # out_dict[fe+"_pred"].append(5)
        i += 1


out_df = pd.DataFrame(out_dict)
for f in pred_feature:
    filename = "./module/models/scalers/scaler_" + f + ".pickle"
    with open(filename, 'rb') as handle:
        scaler = pickle.load(handle)
    out_df[f] = scaler.inverse_transform(out_df[[f]])
    out_df[f+"_pred"] = scaler.inverse_transform(out_df[[f+"_pred"]])

out_df.to_csv("output_test2.csv", index=False)

print(8)
