import pandas as pd

df_concat = pd.read_csv("./module/data/Complete_Data.csv")

df_index = 0


lsh_list = []
def build_percent_1(minute):
    global df_index
    percent_list = []
    if df_index != len(df_concat)-1:
        for c in range(minute + 1):
            percent_list.append(df_concat["Close"][c + df_index])
        percent = (percent_list[1]-percent_list[0])/sum(percent_list)
        df_index += 1
    else:
        percent = df_concat["Close"][2633117]
        # long
    if percent >= 0 :
        lsh_list.append(1)
    else:
        lsh_list.append(0)

    return round(percent, 4)




def build_percent(minute):
    global df_index
    percent_sum = 0
    if df_index != len(df_concat)-minute:
        for c in range(minute+1):
            percent_sum += df_concat["percent_1"][c + df_index]

        df_index += 1
        return round(percent_sum, 4)

df_concat["percent_1"] = df_concat["Close"].apply(lambda x: build_percent_1(minute=1))
df_concat["lsh"] = lsh_list
# min_list_check = [2, 3, 5, 10, 20, 30, 40, 60, 90, 120, 180, 240, 300, 360]
#
# for i in min_list_check:
#     name = "percent_" + str(i)
#     df_index = 0
#     df_concat[name] = df_concat["percent_1"].apply(lambda x: build_percent(minute=i))
#     print(name + "Done ..")

df_concat.dropna(inplace=True)
df_concat.to_csv("./module/data/new_data_persent1_lsh.csv", index=False)
print(8)