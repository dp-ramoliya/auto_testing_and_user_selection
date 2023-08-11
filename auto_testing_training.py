import os
import joblib
import warnings
import psycopg2
import numpy as np
import pandas as pd
from sklearn import metrics
from termcolor import colored
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from utills import db_to_pandas, input_df_wear, calc_rate
from feature_select import feature_select_GB, feature_select_regression

warnings.filterwarnings("ignore")
np.set_printoptions(precision=3, suppress=True)

def df_wear_cleaning(df_wear):
    """
    If wear is zero then delete the row.

    Replacement and maintainance record wear values must not get differenciated from previous record so
    the below line of code will prevent it
    """
    df_wear.loc[df_wear['mill_hours_diff']<0, 'wear_diff'] = df_wear.loc[df_wear['mill_hours_diff']<0, 'wear']
    df_wear.loc[df_wear['mill_hours_diff']<0, 'mill_hours_diff'] = df_wear.loc[df_wear['mill_hours_diff']<0, 'hours']
    df_wear.drop(df_wear[df_wear['wear'] == 0].index, inplace=True)
    df_wear.drop(df_wear[df_wear['mill_hours_diff'] == 0].index , inplace=True)
    return df_wear

def all_asset_pid(asset_ids, conn):
    """Get all the pids for the Assets"""
    sql_query_pid = 'SELECT * FROM t_mill_model_pid_relation WHERE asset_id in {};'.format(asset_ids)
    df_asset_pids = pd.read_sql_query(sql_query_pid, conn)
    return df_asset_pids

conn = psycopg2.connect(
   database="regression", user='postgres', password='root', host='127.0.0.1', port= '5432'
)

# data unit value come from step function
# now use static for testing

data_unit = 3
cursor = conn.cursor()
sql_all_asset = 'SELECT * FROM t_measurement_item_id_relation WHERE unit={}'.format(data_unit)
asset_measurement_id_relation_df = pd.read_sql_query(sql_all_asset, conn)

all_asset_id = tuple(asset_measurement_id_relation_df.asset_id.unique())
all_input=[]
for i in all_asset_id:
    try:
        out_wear, relation = input_df_wear(i, conn)
        all_input.append(out_wear)
    except Exception as e:
        print(e)
        print("Error in input_df_wear function for asset_id: {}".format(i))  

df_wear = pd.concat(all_input, ignore_index=True)

df_wear = df_wear_cleaning(df_wear)

df_wear['wear_per_day'] = (df_wear['wear_diff'] / df_wear['mill_hours_diff'])*24

df_wear = calc_rate(df_wear)

with pd.option_context("mode.use_inf_as_na", True):
    df_wear.loc[
        (df_wear["rate_mill_h"] < 0) | (1 < df_wear["rate_mill_h"]), "rate_mill_h"
    ] = np.nan
    df_wear["finite_rate_mill_h"] = df_wear["rate_mill_h"].notna().astype(float)
    df_wear["rate_mill_h"].fillna(-9999, inplace=True)
    # df_wear["rate_mill_h"].fillna(0, inplace=True)
    df_wear["finite_rate_mill_h"].value_counts()
    
df_asset_pid = all_asset_pid(all_asset_id, conn)
pids_and_name = {i:j for i,j in zip(df_asset_pid['t_pid_no_text'].values, df_asset_pid['sensor_name'].values)}
df_asset_pid.set_index('sensor_name', inplace=True)

pids = tuple(df_asset_pid['t_pid_no_text'].values)

pids = tuple(df_asset_pid['t_pid_no_text'].values)
df_ai = db_to_pandas(conn, "t_iot_data", pids)

col_g = df_asset_pid.loc['supply', 't_pid_no_text']
# df_pids.loc['supply', 't_pid_no_text']
df_ai.loc[:, col_g] = df_ai.loc[:, col_g].mask(df_ai.loc[:, col_g] < 0)

# col_g = pid_df.loc[pid_df["item"] == "HGI", "pid"].values.tolist()
col_g = df_asset_pid.loc['HGI', 't_pid_no_text']
df_ai.loc[:, col_g] = df_ai.loc[:, col_g].mask(
    (df_ai.loc[:, col_g] < 20) | (100 < df_ai.loc[:, col_g])
)

# col_g = pid_df.loc[pid_df["item"] == "moisture", "pid"].values.tolist()
col_g = df_asset_pid.loc['moisture', 't_pid_no_text']
df_ai.loc[:, col_g] = df_ai.loc[:, col_g].mask(
    (df_ai.loc[:, col_g] < 0.1) | (20 < df_ai.loc[:, col_g])
)

df_ai.mask(df_ai > 1e4, inplace=True)
df_ai = df_ai.resample('D').mean()

mld_pipeline_4 = make_pipeline(
        SelectFromModel(estimator=ExtraTreesRegressor(max_features=0.7000000000000001, n_estimators=100), threshold=0.1),
        KNeighborsRegressor(n_neighbors=43, p=2, weights="uniform")
    )
mld_pipeline_5 = make_pipeline(
    SelectFromModel(estimator=ExtraTreesRegressor(max_features=0.8, n_estimators=100), threshold=0.1),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=9, min_samples_leaf=1, min_samples_split=8)),
    RandomForestRegressor(bootstrap=True, max_features=1.0, min_samples_leaf=18, min_samples_split=12, n_estimators=100)
)
mld_pipeline_6 = make_pipeline(
    SelectFromModel(estimator=ExtraTreesRegressor(max_features=0.55, n_estimators=100), threshold=0.3),
    KNeighborsRegressor(n_neighbors=42, p=1, weights="uniform")
)
mld_pipeline_7 = make_pipeline(
    SelectPercentile(score_func=f_regression, percentile=1),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=1, min_samples_leaf=4, min_samples_split=5)),
    ExtraTreesRegressor(bootstrap=True, max_features=0.9500000000000001, min_samples_leaf=11, min_samples_split=4, n_estimators=100)
)
mld_pipeline_8 = make_pipeline(
    StackingEstimator(estimator=LinearSVR(C=0.001, dual=True, epsilon=0.0001, loss="squared_epsilon_insensitive", tol=0.001)),
    RandomForestRegressor(bootstrap=True, max_features=1.0, min_samples_leaf=7, min_samples_split=18, n_estimators=100)
)

model_prediction = []
model_actual = []
model_rmse = []

item_id_relation = "SELECT * FROM t_measurement_item_id_relation"

mdf = pd.read_sql_query(item_id_relation, conn)
mdf = mdf[mdf['unit'] == data_unit]

df_feature_GB = pd.DataFrame(columns=['asset_id', 'msumt_id', 'features'])

"""Generating Pickle file for All Asset ID"""

for m in all_asset_id:
    print("Training of Mill/Asset Started ", m)
    df_wear_asset = df_wear[df_wear["asset_id"]==m]
    longitude = list(df_wear_asset['measurement_item_id'].unique())
    model_list = [mld_pipeline_4,mld_pipeline_5,mld_pipeline_6,mld_pipeline_7,mld_pipeline_8]*3
    model_pipe_dict = dict(map(lambda i,j : (i,j) , longitude,model_list))
    all_sensor_id = df_asset_pid.loc[df_asset_pid['asset_id']== m]
    dict_sensor_dfs = df_ai[all_sensor_id['t_pid_no_text']]

    dict_sensor_dfs.dropna(inplace=True)
    for r in df_wear_asset.measurement_item_id.unique():
        print("logitude ", r)
        dict_sensor_dfs = dict_sensor_dfs[:df_wear_asset.date[df_wear_asset['measurement_item_id']==r].max()]
        loop_df_m = df_wear_asset[df_wear['measurement_item_id']==r]
        sensor_df_list = []

        for i in loop_df_m.index:
            sensor_df_loop = dict_sensor_dfs[loop_df_m.loc[i,'date_start']:loop_df_m.loc[i, 'date']]
            sensor_df_loop.dropna(inplace=True)
            sensor_df_loop['{}_{}_rate_mill_h'.format(data_unit,r)]= loop_df_m.loc[i, 'rate_mill_h']
            sensor_df_loop['{}_{}_wear_rate'.format(data_unit,r)] = loop_df_m.loc[i, 'wear_per_day']
            sensor_df_list.append(sensor_df_loop)

        sensor_data_with_wear = pd.concat(sensor_df_list)

        inf_index = sensor_data_with_wear[sensor_data_with_wear['{}_{}_wear_rate'.format(data_unit,r)]==np.inf].index
        sensor_data_with_wear.drop(inf_index, inplace=True)
        sensor_data_with_wear.dropna(inplace=True)
        sensor_data_with_wear.replace(to_replace = -9999, value = 0, inplace=True)
        
        # Splitting the dataset into the Training set and Test set
        X_temp = sensor_data_with_wear.iloc[:,:7]
        y = sensor_data_with_wear.iloc[:,-1:]
        x_column = feature_select_GB(X=X_temp, y=y, asset_id=m, msumt_id=r, pids_and_name=pids_and_name)
        df_feature_GB.loc[len(df_feature_GB.index)] = [m, r, x_column] 
        
        X = sensor_data_with_wear[x_column]
        print(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
        
        mlr_loop = model_pipe_dict[r].fit(X_train, y_train)

        y_pred= mlr_loop.predict(X_test)
        model_actual.append(y_test.sum().values[0])
        model_prediction.append(sum(y_pred))
        rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        model_rmse.append(rootMeanSqErr)
        
        if not os.path.exists('./auto_training'):
            os.mkdir('./auto_training')
        joblib.dump(mlr_loop, 'auto_training/model_{}.pkl'.format(r))

        print('rootMeanSqErr_of_{}_{} : {}'.format(data_unit, r, rootMeanSqErr))

if not os.path.exists('./data'):
    os.mkdir('./data')
df_feature_GB.to_csv("data/features_GB.csv", index=False)
