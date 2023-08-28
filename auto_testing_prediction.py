import os
import json
import joblib
import ast
import psycopg2
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utills import get_threshold_value, input_df_wear, getCurrentSensorCount,df_wear_cleaning, \
                   calc_rate, all_asset_pid, db_to_pandas, scale_mill, scale_mill_state, supply_current_convert, \
                   estimate_operation_rate, sim_days_calculate

np.set_printoptions(precision=3, suppress=True)
warnings.filterwarnings("ignore")

conn = psycopg2.connect(
   database="regression", user='postgres', password='root', host='127.0.0.1', port= '5432'
)

data_unit = 5
input_json = """{
    "asset-id": 11609,
    "measurement-item-set-id": 1111,
    "threshold": 65,
    "measurement-record-set-ids": []
}"""

input_json = json.loads(input_json)
asset_id = int(input_json['asset-id'])
msumt_item_set_id = int(input_json['measurement-item-set-id'])

cursor = conn.cursor()

# Selecting Threshold Values from DB
threshold_wear = get_threshold_value(conn, asset_id)

# Selecting the data from DB for the asset_id
df_wear, relation = input_df_wear(asset_id, conn)
print("df_wear created.")

highest_date_row = df_wear.sort_values(by='date').iloc[-1]
highest_date = str(highest_date_row['date'])
highest_hours = highest_date_row['hours']

current_sensor_count = getCurrentSensorCount(conn, asset_id, highest_date)
print("current_sensor_count: ", current_sensor_count)
# clean the df_wear and remove hours = 1 records, negative value records and wear = 0 records
df_wear = df_wear_cleaning(df_wear)

# Create a wear_rate per day column
df_wear = calc_rate(df_wear)

# filter rate mill hours and finite rate mill hours column
with pd.option_context("mode.use_inf_as_na", True):
    df_wear.loc[
        (df_wear["rate_mill_h"] < 0) | (1 < df_wear["rate_mill_h"]), "rate_mill_h"
    ] = np.nan
    df_wear["finite_rate_mill_h"] = df_wear["rate_mill_h"].notna().astype(float)
    df_wear["rate_mill_h"].fillna(-9999, inplace=True)
    #df_wear["rate_mill_h"].fillna(0, inplace=True)
    df_wear["finite_rate_mill_h"].value_counts()

# Fetching pid relation from database to dataframe
df_pids = all_asset_pid(asset_id, conn)
pids_and_name = {i:j for i,j in zip(df_pids['t_pid_no_text'].values, df_pids['sensor_name'].values)}
df_pids.set_index('sensor_name', inplace=True)
pids = tuple(df_pids['t_pid_no_text'].values)

# Convert Database to Panda's Dataframe
df_ai = db_to_pandas(conn, "t_iot_data", pids)

# Filtering Dataframe which is Graterthan 10000. and convert from Hour basis Data to Day wise
df_ai.mask(df_ai > 1e4, inplace=True)
df_ai = df_ai.resample('D').mean()

col_g = df_pids.loc['supply', 't_pid_no_text']
df_ai.loc[:, col_g] = df_ai.loc[:, col_g].mask(df_ai.loc[:, col_g] < 0)

col_g = df_pids.loc['HGI', 't_pid_no_text']
df_ai.loc[:, col_g] = df_ai.loc[:, col_g].mask(
    (df_ai.loc[:, col_g] < 20) | (100 < df_ai.loc[:, col_g])
)

col_g = df_pids.loc['moisture', 't_pid_no_text']
df_ai.loc[:, col_g] = df_ai.loc[:, col_g].mask(
    (df_ai.loc[:, col_g] < 0.1) | (20 < df_ai.loc[:, col_g])
)

# データが全くない時点が最後に続く場合は除く
with_data = df_ai.index[df_ai.notna().sum(axis=1) > 0]
t1 = with_data.min()
t2 = with_data.max()
df_ai = df_ai.loc[t1:t2, :]

# Analog Inputとの紐づけ
df_wear["wear_start"] = df_wear["wear"] - df_wear["wear_diff"]
df_wear["start_index"] = df_ai.index.get_indexer(df_wear["date_start"].values) + 1
df_wear["end_index"] = df_ai.index.get_indexer(df_wear["date"].values) + 1


df_wear["rate_mill_h"].fillna(value= 0, inplace=True)
# Train-test split

df_wear_train = df_wear.copy()
df_wear_test = []
for i, df in df_wear.groupby(["measurement_item_id"]):
    df_wear_test.append(df.loc[df["date"] == df["date"].max(), :].copy())

df_wear_test = pd.concat(df_wear_test)
df_wear_test.loc[:, "end_index"] = len(df_ai)

# prepare simulation data frame now
df_wear_sim = []
for r in np.arange(0.1, 1.1, 0.1):
    df_append = df_wear_test.copy()
    df_append["rate_mill_h"] = r
    df_append["finite_rate_mill_h"] = 1.0
    df_wear_sim.append(df_append)

df_wear_sim = pd.concat(df_wear_sim, axis=0)

# calculate utilization part
test_start = df_ai.index.max()
vnames_minmax = [[df_pids.loc["supply", "t_pid_no_text"]],
    [df_pids.loc["current", "t_pid_no_text"]]]

scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
state_minmax, state_minmax_notna = scale_mill(vnames_minmax, df_ai, scaler, test_start)

vnames_std = [
    [df_pids.loc["HGI", "t_pid_no_text"]],
    [df_pids.loc["moisture", "t_pid_no_text"]],
    [df_pids.loc["air_pressure", "t_pid_no_text"]],
    [df_pids.loc["vibration", "t_pid_no_text"]],
]

scaler = StandardScaler()
state_std = scale_mill_state(vnames_std, df_ai, scaler)
state_std_notna = (~np.isnan(state_std)).astype(float)
np.nan_to_num(state_std, copy=False, nan=-9999)
# np.nan_to_num(state_std, copy=False, nan=0.0)

# データ整形
df_wear_train.sort_values(
    ["measurement_item_id", "date"], inplace=True
)
df_wear_test.sort_values(
    ["measurement_item_id", "date"], inplace=True
)
df_wear_sim.sort_values(
    ["measurement_item_id", "date"], inplace=True
)

data = dict(
    N=len(df_wear_train),
    N_test=len(df_wear_test),
    N_sim=len(df_wear_sim),
    # length of time series
    T=len(df_ai),
    delta_t=1,
    # number of variables in time series
    V_minmax=state_minmax.shape[2],
    V_std=state_std.shape[2],
    state_minmax_raw=state_minmax,
    state_minmax_notna_raw=state_minmax_notna,
    state_std_raw=state_std,
    state_std_notna_raw=state_std_notna,
    # number of mill
    M=1,
    # number of position
    P=5,
    # start and end of wear data for each position
    #i_pos=pos_train,
    wear_target=df_wear_train["wear"].values,
    # 1 wear_raw, 2 operation rate, 3 finite flag of operation rate
    wear=df_wear_train.loc[
        :, ["wear", "rate_mill_h", "finite_rate_mill_h"]
    ].values,
    #mill=df_wear_train["mill_index"].values,
    mill=0,
    t=df_wear_train[["start_index", "end_index"]].values,
    #i_pos_test=pos_test,
    # for RUL prediction, input should be "wear" instead of "wear_start"
    wear_test=df_wear_test.loc[:, ["wear", "rate_mill_h", "finite_rate_mill_h"]].values,
    mill_test=0,
    #mill_test=df_wear_test["mill_index"].values,
    t_test=df_wear_test[["start_index", "end_index"]].values,
    #i_pos_sim=pos_sim,
    # for RUL prediction, input should be "wear" instead of "wear_start"
    wear_sim=df_wear_sim.loc[:, ["wear", "rate_mill_h", "finite_rate_mill_h"]].values,
    mill_sim=0,
    #mill_sim=df_wear_sim["mill_index"].values,
    t_sim=df_wear_sim[["start_index", "end_index"]].values,
)

state_minmax=data["state_minmax_raw"]
state_minmax_notna = data['state_minmax_notna_raw']

N = data["N"]
mill = data["mill"]
t = data["t"]
V_std = data["V_std"]

term_supply, term_current, N_tp, supply, term_supply_na, term_current_na = supply_current_convert(N, mill, t, state_minmax, state_minmax_notna, state_std, state_std_notna, V_std)
supply_rate = supply / N_tp

# For test Data
N_test = data["N_test"]
mill_test = data["mill_test"]
t_test = data["t_test"]

term_supply_test, term_current_test, N_tp_test, supply_test, term_supply_test_na, term_current_test_na = supply_current_convert(N_test, mill_test, t_test, state_minmax, state_minmax_notna, state_std, state_std_notna, V_std)
# supply_rate_test = supply_test / N_tp_test

# For Sim Data
N_sim = data["N_sim"]
mill_sim = data["mill_sim"]
t_sim = data["t_sim"]

term_supply_sim, term_current_sim, N_tp_sim, supply_sim, term_supply_sim_na, term_current_sim_na = supply_current_convert(N_sim, mill_sim, t_sim, state_minmax, state_minmax_notna, state_std, state_std_notna, V_std)
# supply_rate_sim = supply_sim / N_tp_sim

# Normalization
for v in range(0, (V_std - 2)):
    N_v = sum(term_supply_na[:, v])
    mean_v = np.dot(term_supply[:, v], term_supply_na[:, v]) / N_v
    sd_v = np.dot((term_supply[:, v] - mean_v)*(term_supply[:, v] - mean_v), term_supply_na[:, v]) / (N_v - 1)
    if (sd_v > 0):
        term_supply[:, v] = (term_supply[:, v] - mean_v) / sd_v
        term_supply[:, v] *= term_supply_na[:, v]
    else:
        term_supply[:, v] =np.full(N,0)
    if (sd_v > 0):
        term_supply_test[:, v] = (term_supply_test[:, v] - mean_v) / sd_v
        term_supply_test[:, v] *= term_supply_test_na[:, v]
    else:
        term_supply_test[:, v] = np.full(N_test,0)
    if (sd_v > 0):
        term_supply_sim[:, v] = (term_supply_sim[:, v] - mean_v) / sd_v
        term_supply_sim[:, v] *= term_supply_sim_na[:, v]
    else:
        term_supply_sim[:, v] = np.full(N_test,0)

print("normalization of term supply completed.")

for v in range(2):
    N_v = sum(term_current_na[:, v])
    mean_v = np.dot(term_current[:, v], term_current_na[:, v]) / N_v
    sd_v = np.dot((term_current[:, v] - mean_v)*(term_current[:, v] - mean_v), term_current_na[:, v]) / (N_v - 1)
    if (sd_v > 0):
        term_current[:, v] = (term_current[:, v] - mean_v) / sd_v
        term_current[:, v] *= term_current_na[:, v]
    else:
        term_current[:, v] =np.full(N,0)
    if (sd_v > 0):
        term_current_test[:, v] = (term_current_test[:, v] - mean_v) / sd_v
        term_current_test[:, v] *= term_current_test_na[:, v]
    else:
        term_current_test[:, v] = np.full(N_test,0)
    if (sd_v > 0):
        term_current_sim[:, v] = (term_current_sim[:, v] - mean_v) / sd_v
        term_current_sim[:, v] *= term_current_sim_na[:, v]
    else:
        term_current_sim[:, v] = np.full(N_sim,0)

print("normalization of term current completed")

# Finding Slop and intercept
est_current_operation_rate = estimate_operation_rate(data["wear"], supply_test, N_tp_test, supply_rate)
df_wear_test.loc[:, "Utilization"] = abs(est_current_operation_rate)
utilization_dict = dict(zip(df_wear_test.measurement_item_id, df_wear_test.Utilization))

# Prediction for test dataframe
print("Prediction srarted for test dataframe")

rm_days_test = []
rm_days_sim = []
total_hours_list = []

feature_GB = pd.read_csv(f"data/features_GB_{data_unit}.csv")
feature_GB = feature_GB[feature_GB['asset_id'] == asset_id]
feature_GB['features'] = feature_GB['features'].apply(ast.literal_eval)

for r in df_wear.measurement_item_id.unique():
    last_date = df_wear.loc[(df_wear['measurement_item_id'] == r),['date']].max().values
    
    dict_sensor_dfs = df_ai[last_date[0]:]
    dict_sensor_dfs.dropna(inplace=True)
    model_loop = joblib.load(open('auto_training/model_{}.pkl'.format(r),'rb'))
    
    print("Prediction start of Longitude : ", r)
    rate_mill_h = df_wear_test[df_wear_test['measurement_item_id'] ==r].rate_mill_h.values[0]
    total_wear = df_wear_test[df_wear_test['measurement_item_id']==r].wear.values[0]

    supply = dict_sensor_dfs[df_pids.loc['supply', 't_pid_no_text']].mean()
    current = dict_sensor_dfs[df_pids.loc['current', 't_pid_no_text']].mean()
    HGI = dict_sensor_dfs[df_pids.loc['HGI', 't_pid_no_text']].mean()
    moisture = dict_sensor_dfs[df_pids.loc['moisture', 't_pid_no_text']].mean()
    air_pressure = dict_sensor_dfs[df_pids.loc['air_pressure', 't_pid_no_text']].mean()
    vibration = dict_sensor_dfs[df_pids.loc['vibration', 't_pid_no_text']].mean()

    sensor_dict = {
        "supply":supply,
        "current":current,
        "HGI":HGI,
        "moisture":moisture,
        "air_pressure":air_pressure,
        "vibration":vibration,
        "rate_mill_h":rate_mill_h
    }

    f_value = feature_GB[feature_GB['msumt_id']==r]['features'].explode().tolist()
    name_change = [pids_and_name[i] if pids_and_name.get(i) else i for i in f_value]
    name_change = ['rate_mill_h' if 'rate_mill_h' in j else j for j in name_change ]
    f_list = [sensor_dict[i] if sensor_dict.get(i) else i for i in name_change]
    
    calc_wear = model_loop.predict([f_list])
    calc_wear = calc_wear.reshape(1)
    model_calc_rate = calc_wear[0]

    rm_days_sim.extend(sim_days_calculate(threshold_wear, total_wear, model_calc_rate, rate_mill_h, utilization_dict[r]))

    r_days = (threshold_wear-total_wear)/model_calc_rate
    
    if r_days > 5475:
        r_days = 5475

    rm_days_test.append(int(r_days))
    remaining_hours = int(r_days)*24

    today_date = datetime.now()
    date_1 = pd.to_datetime(last_date[0])
    
    end_date = date_1 + timedelta(days=int(r_days))
    
    if end_date <= today_date:
        total_hours =  highest_hours  + remaining_hours
        
    else:
        predict_hours = (end_date - today_date)/ pd.Timedelta(hours=1)
        total_hours =  highest_hours  + current_sensor_count + int(predict_hours)

    total_hours_list.append(total_hours)
    print("Prediction end of Longitude : ", r)


print("Threshold Date Generates for All Roller.")

df_wear_test.loc[:, "remain_d"] = rm_days_test
df_wear_test.loc[:, "Asset_id"] = asset_id
df_wear_test.loc[:, "Total_Hours"] = total_hours_list
df_wear_test.loc[:, "threshold_date"] = pd.to_datetime(df_wear_test["date"].dt.strftime('%Y-%m-%d')) + pd.to_timedelta(
    df_wear_test.loc[:, "remain_d"].values, unit="D"
)
df_wear_test.loc[:, "threshold_date"] = df_wear_test.loc[:, "threshold_date"].dt.floor("D")

#, errors='coerce', format='%Y-%m-%d'
df_wear_sim.loc[:, "remain_d"] = rm_days_sim
df_wear_sim["remain_d"] = df_wear_sim.loc[:, "remain_d"].apply(lambda x: 25000 if x > 25000 else x)
df_wear_sim.loc[:, "threshold_date"] = pd.to_datetime(df_wear_sim["date"].dt.strftime('%Y-%m-%d')) + pd.to_timedelta(
    df_wear_sim.loc[:, "remain_d"].values, unit="D"
    )
df_wear_sim.loc[:, "threshold_date"] = df_wear_sim.loc[:, "threshold_date"].dt.floor("D")


df_test_out = df_wear_test[["measurement_item_id", "Utilization", "threshold_date", "Total_Hours"]]
df_wear_sim = df_wear_sim[["measurement_item_id","rate_mill_h", "threshold_date"]]

df_sim_out = df_wear_sim.pivot(index='measurement_item_id', columns='rate_mill_h', values='threshold_date')
df_sim_out.columns = np.array(["稼働率 "], dtype=object) + df_sim_out.columns.values.round(1).astype(str)
df_test_out.set_index(['measurement_item_id'], inplace=True)

df_out = pd.concat([df_test_out, df_sim_out], axis=1, sort=False)

df_out.rename(columns={'threshold_date':'寿命到達日'}, inplace=True)

hour_col = ["残時間"] + (df_sim_out.columns.values + np.array(" 残時間")).tolist()
date_col = ["寿命到達日"] + df_sim_out.columns.values.tolist()

date_now = pd.Timestamp.now()
df_out.loc[:, hour_col] = ((df_out[date_col] - date_now).values / pd.Timedelta(1, "h")).round()
df_out.reset_index(inplace=True)
df_out = relation.merge(df_out, on='measurement_item_id', how='left')
df_out.rename(columns={'threshold_date':'寿命到達日',
            'longitude': '軸方向位置',
            'Utilization':'推定稼働率',
            'Total_Hours':'合計時間'}, inplace=True)

df_out = df_out[[
    'asset_id', 'measurement_item_id', '軸方向位置', '推定稼働率', '寿命到達日', '合計時間', 
    '稼働率 0.1', '稼働率 0.2', '稼働率 0.3', '稼働率 0.4', '稼働率 0.5','稼働率 0.6', 
    '稼働率 0.7', '稼働率 0.8', '稼働率 0.9', '稼働率 1.0', '残時間', '稼働率 0.1 残時間',
    '稼働率 0.2 残時間', '稼働率 0.3 残時間', '稼働率 0.4 残時間', '稼働率 0.5 残時間', 
    '稼働率 0.6 残時間', '稼働率 0.7 残時間', '稼働率 0.8 残時間', '稼働率 0.9 残時間', '稼働率 1.0 残時間'
    ]]
# last_input = df_ai.index.max()

try:
    select_last_date_train = """SELECT created_date, model_rmse, avg_rmse FROM t_model_setsubi_training_data_check
                                WHERE asset_id = '{}' ORDER BY created_date DESC LIMIT 1""".format(asset_id)

    cursor.execute(select_last_date_train)
    db_value = cursor.fetchall()[0]
    last_date_train = db_value[0]
    last_date_train = last_date_train.strftime("%Y-%m-%d %H:%M:%S")
    rmse = db_value[1]
    rmse_operation = db_value[2]


except Exception as e:
    print(e)
    print("Error While Selecting Last Date Train")
    last_date_train = "Not Found"
    rmse = "Not Found"
    rmse_operation = "Not Found"


setting = [
    "# asset-id :{}".format(asset_id),
    "# measurement-item-set-id : {}".format(msumt_item_set_id),
    "# Last-train-date : {}".format(last_date_train),
    "# 摩耗推定の平均二乗誤差平方根(RMSE mm), {}, 20 mm を超える場合は、余寿命予測の信頼性に注意が必要です。".format(rmse),
    "# 稼働率推定の平均二乗誤差平方根(RMSE), {}, 0.01を超える場合は、余寿命予測の信頼性に注意が必要です。\n".format(rmse_operation),
]

fn_out_local = "output/u{}_{}.csv".format(
    str(asset_id), datetime.now().strftime("%Y%m%d%H%M%S")
)
if not os.path.exists('./output'):
    os.mkdir('./output')

with open(fn_out_local, mode="w", encoding="utf-8-sig") as f:
    f.writelines("\n".join(setting))
    df_out.to_csv(f, index=False, mode="a")

conn.close()  # close connection
