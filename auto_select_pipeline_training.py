import os
import joblib
import psycopg2
import warnings
import numpy as np
import pandas as pd
from math import sqrt
from termcolor import colored
from sklearn.linear_model import Lasso
from datetime import timedelta, datetime
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
warnings.filterwarnings("ignore")
np.set_printoptions(precision=3, suppress=True)

from utills import db_to_pandas, input_df_wear, calc_rate

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

create_dir = ['rmse', 'features', 'dynamic_train']
for d in create_dir:
    if not os.path.exists(f'./{d}'):
        os.mkdir(f'./{d}')

conn = psycopg2.connect(
   database="regression", user='postgres', password='root', host='127.0.0.1', port= '5432'
)

# ON STG
# 6464/1117
# 8980/1123
# 11609/1111

# data unit value come from step function
# now use static for testing
data_unit = 4
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
        print(f"Error in input_df_wear function for asset_id: {i}")

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
    #df_wear["rate_mill_h"].fillna(0, inplace=True)
    df_wear["finite_rate_mill_h"].value_counts()


df_asset_pid = all_asset_pid(all_asset_id, conn)
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

model_prediction = []
model_actual = []
model_rmse = []

master_rmse = {}
master_feature = {}
try:
    for m in all_asset_id:
        print("Training of Mill/Asset Started ", m)
        df_wear_asset = df_wear[df_wear["asset_id"]==m]
        longitude = list(df_wear_asset['measurement_item_id'].unique())
        
        all_sensor_id = df_asset_pid.loc[df_asset_pid['asset_id']== m]
        dict_sensor_dfs = df_ai[all_sensor_id['t_pid_no_text']]
        
        dict_sensor_dfs.dropna(inplace=True)
        
        for r in df_wear_asset.measurement_item_id.unique():
            dict_sensor_dfs = dict_sensor_dfs[:df_wear_asset.date[df_wear_asset['measurement_item_id']==r].max()]
            print(colored(r, 'blue', attrs=['bold']), "--",df_wear_asset.date[df_wear_asset['measurement_item_id']==r].max())

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

            X = sensor_data_with_wear.iloc[:,:7]
            y = sensor_data_with_wear.iloc[:,-1:]

            feature_selector = RandomForestRegressor(n_estimators=100)
            feature_selector.fit(X, y)
            important_features = X.columns[feature_selector.feature_importances_ > 0.05]

            # L1 Regularization (Lasso) 
            # feature_selector = SelectFromModel(Lasso(alpha=0.05)) 
            # feature_selector.fit(X, y) 
            # important_features = X.columns[feature_selector.get_support()]

            print("important_features")
            print(important_features)
            master_feature[str(r)] = list(important_features)
            feature_selected = df_asset_pid[df_asset_pid['t_pid_no_text'].isin(list(important_features))]
            print("*"*50)
            print(feature_selected)


            models = [
            ("randomforestregressor", RandomForestRegressor()),
            ("gradientboostingregressor", GradientBoostingRegressor()),
            ("extratreesregressor", ExtraTreesRegressor()),
            # ("kneighborsregressor", KNeighborsRegressor()),
            # ("decisiontreeregressor", DecisionTreeRegressor()),
            ]

            best_model = None
            best_rmse = float("inf")

            for model_name, model in models:
                pipeline = make_pipeline(SelectFromModel(feature_selector), model)

                # Hyperparameter optimization using GridSearchCV
                # Hyperparameter optimization using GridSearchCV
                param_grid = {
                    f"{model_name}__n_estimators": [100, 200, 300]
                }

                if model_name in [
                    "randomforestregressor",
                    "gradientboostingregressor",
                    "decisiontreeregressor",
                ]:
                    param_grid[f"{model_name}__max_depth"] = [None, 5, 10]

                grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="neg_mean_squared_error")
                grid_search.fit(X[important_features], y)

                # Best model from grid search
                current_model = grid_search.best_estimator_

                # Model Evaluation
                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X[important_features], y, test_size=0.2, random_state=42)

                # Train the current model on the training set
                current_model.fit(X_train, y_train)

                # Use the current model to make predictions on the test set
                y_pred = current_model.predict(X_test)

                # Calculate the root mean squared error (RMSE)
                current_rmse = sqrt(mean_squared_error(y_test, y_pred))

                # Update the best model if the current model has a lower RMSE
                if current_rmse < best_rmse:
                    best_model = current_model
                    best_rmse = current_rmse

            master_rmse[str(r)] = best_rmse
            model_rmse.append(best_rmse)
            
            joblib.dump(current_model, 'dynamic_train/model_{}.pkl'.format(r))
            
            print("Best Model:", colored(best_model, 'green', attrs=['bold'] ))
            print("Best RMSE:",colored(best_rmse, 'red', attrs=['bold']))
            print('rootMeanSqErr_of_{}_{} : {}'.format(data_unit,r,best_rmse))
            print("#"*50)

except KeyboardInterrupt:
    pass

master_feature_df = pd.DataFrame(master_feature.items())
master_feature_df.to_csv(f'features/{data_unit}_master_feature.csv', index=False)

master_rmse_df = pd.DataFrame(master_rmse.items(), columns=["measurement_item_id", "RMSE"])
#master_rmse_df["RMSE"] = master_rmse_df["RMSE"].map(lambda x: "{:.20f}".format(x))
master_rmse_df.to_csv('rmse/{}_master_rmse.csv'.format(data_unit), index=False)
print(master_rmse_df)
