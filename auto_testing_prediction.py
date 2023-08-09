import json
import joblib
import psycopg2
import warnings
import numpy as np
import pandas as pd
from termcolor import colored
from datetime import timedelta, datetime
from numpy.core.umath_tests import inner1d
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utills import get_threshold_value, input_df_wear, getCurrentSensorCount,df_wear_cleaning, \
                   calc_rate, all_asset_pid, db_to_pandas, scale_mill, scale_mill_state, supply_current_convert, \
                   estimate_operation_rate, sim_days_calculate
np.set_printoptions(precision=3, suppress=True)
warnings.filterwarnings("ignore")

conn = psycopg2.connect(
   database="regression", user='postgres', password='root', host='127.0.0.1', port= '5432'
)

input_json = """{
    "asset-id": 8980,
    "measurement-item-set-id": 1123,
    "threshold": 65,
    "measurement-record-set-ids": []
}"""

input_json = json.loads(input_json)
asset_id = int(input_json['asset-id'])
print(asset_id, type(asset_id))
cursor = conn.cursor()
threshold_wear = get_threshold_value(conn, asset_id)

# Selecting the data from DB for the asset_id
df_wear, relation = input_df_wear(asset_id, conn)
print("df_wear after concat", df_wear)

highest_date_row = df_wear.sort_values(by='date').iloc[-1]
highest_date = str(highest_date_row['date'])
highest_hours = highest_date_row['hours']

print("highest_date: ", highest_date)
print("highest_hours: ", highest_hours)
current_sensor_count = getCurrentSensorCount(conn,asset_id,highest_date)
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
    # df_wear["rate_mill_h"].fillna(0, inplace=True)
    df_wear["finite_rate_mill_h"].value_counts()

# Fetching pid relation from database to dataframe
df_pids = all_asset_pid(asset_id, conn)
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

df_wear_train = df_wear.copy()

df_wear_test = []
for i, df in df_wear.groupby(["measurement_item_id"]):
    df_wear_test.append(df.loc[df["date"] == df["date"].max(), :].copy())

df_wear_test = pd.concat(df_wear_test)
df_wear_test.loc[:, "end_index"] = len(df_ai)

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
#np.nan_to_num(state_std, copy=False, nan=0.0)

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
    # i_pos=pos_train,
    wear_threshold=65,
    wear_target=df_wear_train["wear"].values,
    # 1 wear_raw, 2 operation rate, 3 finite flag of operation rate
    wear=df_wear_train.loc[
        :, ["wear", "rate_mill_h", "finite_rate_mill_h"]
    ].values,
    # mill=df_wear_train["mill_index"].values,
    mill=0,
    t=df_wear_train[["start_index", "end_index"]].values,
    # i_pos_test=pos_test,
    # for RUL prediction, input should be "wear" instead of "wear_start"
    wear_test=df_wear_test.loc[:, ["wear", "rate_mill_h", "finite_rate_mill_h"]].values,
    mill_test=0,
    # mill_test=df_wear_test["mill_index"].values,
    t_test=df_wear_test[["start_index", "end_index"]].values,
    # i_pos_sim=pos_sim,
    # for RUL prediction, input should be "wear" instead of "wear_start"
    wear_sim=df_wear_sim.loc[:, ["wear", "rate_mill_h", "finite_rate_mill_h"]].values,
    mill_sim=0,
    # mill_sim=df_wear_sim["mill_index"].values,
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
print("normalization of term supply completed")

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

data_unit = 4
# df_rmse = pd.read_csv('rmse/{}_master_rmse.csv'.format(data_unit))

def find_closest(lst, k):
    lst.sort()
    closest = lst[0]
    for num in lst:
        if abs(num - k) < abs(closest - k):
            closest = num
        if num > k:
            break
    return closest

def find_wear(lst_d, lst_w, p_date):
    for i,j in zip(lst_d, lst_w):
        if i == p_date:
            return j

def auto_test_df(df_wear):
    auto_df = pd.DataFrame(columns=['measurement_item_id', 'date_l', 'wear_l'])
    msumt_id = []
    date_l = []
    wear_l = []
    rate_mill_h = []

    for i,j in df_wear.groupby('measurement_item_id'):
        j['date'] = j['date'].dt.strftime('%Y-%m-%d')
        date_of_replace = j[j['wear']==0].date.values
        filtered_df = j[(j['date'] > date_of_replace[-2]) & (j['date'] < date_of_replace[-1])]
        print(i, filtered_df.date.values)
        msumt_id.append(i)
        date_l.append(filtered_df.date.to_list())
        wear_l.append(filtered_df.wear.to_list())
        # rate_mill_h.append(filtered_df.wear.to_list())

    auto_df['measurement_item_id'] = msumt_id
    auto_df['date_l'] = date_l
    auto_df['wear_l'] = wear_l
    # auto_df['rate_mill_h'] = rate_mill_h

    auto_df['date_l'] = auto_df.apply(lambda x: [datetime.strptime(i, '%Y-%m-%d') for i in x['date_l']], axis=1)
    auto_df['start_date'] = auto_df.apply(lambda x: x['date_l'][0] + timedelta(days=365*3), axis=1)
    auto_df['from_pred_date'] = auto_df.apply(lambda x: find_closest(x['date_l'] , x['start_date']), axis=1)
    auto_df['total_wear'] = auto_df.apply(lambda x: find_wear(x['date_l'] ,x['wear_l'], x['from_pred_date']), axis=1)
    auto_df.drop(['start_date'], axis=1, inplace=True)

    return auto_df

automation_master_df = auto_test_df(df_wear=df_wear)
# automation_master_df.to_csv("automation_master_df.csv")
pid_id_list = pids

asset_rmse =[]
actual_date = []
rm_days_test = []
total_hours_list = []
remaining_hours_list = []
threshold_wear_list = []

for r in df_wear.measurement_item_id.unique():
    print("prediction of Longitude: ", colored(str(r), 'red', attrs=['bold']))
    last_date = automation_master_df[automation_master_df["measurement_item_id"]==r].from_pred_date.to_list()[0]
    actual_date.append(automation_master_df[automation_master_df["measurement_item_id"]==r].date_l.to_list()[0][-1])
    print("last_date ##",last_date)
    dict_sensor_dfs = df_ai[last_date:]
    dict_sensor_dfs.dropna(inplace=True)
    model_loop = joblib.load(open('auto_training/model_{}.pkl'.format(r),'rb'))
    
    #rate_mill_h = df_wear_test[df_wear_test['measurement_item_id'] ==r].rate_mill_h.values[0]
    #rate_mill_h = df_wear[df_wear['measurement_item_id'] ==r].rate_mill_h.values[0]
    rate_mill_h = df_wear[df_wear['measurement_item_id'] == r][df_wear['date'] == last_date]['rate_mill_h'].values[0]
    threshold_wear = automation_master_df[automation_master_df["measurement_item_id"]==r].wear_l.to_list()[0][-1]
    #threshold_wear = 65
    total_wear = automation_master_df[automation_master_df["measurement_item_id"]==r].total_wear.to_list()[0]
    print("threshold_wear :", threshold_wear)
    print("total_wear :", total_wear)
    print("rate_mill_h: ", rate_mill_h)
    threshold_wear_list.append(threshold_wear)
    hours = df_wear_test[df_wear_test['measurement_item_id']==r].hours.values[0]
    
    supply = dict_sensor_dfs[df_pids.loc['supply', 't_pid_no_text']].mean()
    current = dict_sensor_dfs[df_pids.loc['current', 't_pid_no_text']].mean()
    HGI = dict_sensor_dfs[df_pids.loc['HGI', 't_pid_no_text']].mean()
    moisture = dict_sensor_dfs[df_pids.loc['moisture', 't_pid_no_text']].mean()
    air_pressure = dict_sensor_dfs[df_pids.loc['air_pressure', 't_pid_no_text']].mean()
    vibration = dict_sensor_dfs[df_pids.loc['vibration', 't_pid_no_text']].mean()
    
    calc_wear = model_loop.predict([[supply, current, HGI, moisture, air_pressure, vibration, rate_mill_h]])
    calc_wear = calc_wear.reshape(1)
    model_calc_rate = calc_wear[0]
    print("model_calc_rate ---", model_calc_rate)
    r_days = (threshold_wear-total_wear)/model_calc_rate
    print("$$$$", r_days)
    if r_days > 2190:
        r_days = 2190

    rm_days_test.append(int(r_days))
    remaining_hours = int(r_days)*24

    today_date = datetime.now()
    date_1 = pd.to_datetime(last_date)
    
    end_date = date_1 + timedelta(days=int(r_days))
    
    if end_date <= today_date:
        total_hours =  highest_hours  + remaining_hours
        
    else:
        predict_hours = (end_date - today_date)/ pd.Timedelta(hours=1)
        total_hours =  highest_hours  + current_sensor_count + int(predict_hours)
        
    total_hours_list.append(total_hours)
    remaining_hours_list.append(remaining_hours)

automation_master_df['remain_d'] = rm_days_test
automation_master_df['threshold_wear'] = threshold_wear_list
automation_master_df["actual_date"] = actual_date
automation_master_df.loc[:, "predicted_date"] = pd.to_datetime(automation_master_df["from_pred_date"]) + pd.to_timedelta(
    automation_master_df.loc[:, "remain_d"].values, unit="D"
)
automation_master_df['actual_date'] = pd.to_datetime(automation_master_df['actual_date'])
automation_master_df['predicted_date'] = pd.to_datetime(automation_master_df['predicted_date'])

automation_master_df['diff_days'] = (automation_master_df['actual_date'] - automation_master_df['predicted_date']).dt.days
automation_master_df['flag'] = automation_master_df['diff_days'].apply(lambda x: True if -365 <= x <= 1460 else False)
automation_master_df.drop(['wear_l', 'date_l'], axis=1, inplace=True)
automation_master_df = automation_master_df[['measurement_item_id', 'from_pred_date', 
                                             'total_wear', 'threshold_wear', 'actual_date', 
                                             'predicted_date', 'flag']]
#'remain_d', 'diff_days'
print(automation_master_df)

true_count = automation_master_df['flag'].sum()
total_count = len(automation_master_df)
successful_rate = true_count / total_count * 100  # Calculate the successful rate as a percentage

print(f"Successful Rate of Model: {successful_rate:.2f}%")

automation_master_df.to_csv(f'review/{asset_id}_auto_predicted_test.csv', index=False)
