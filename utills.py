import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from numpy.core.umath_tests import inner1d

def db_to_pandas(client, MEASUREMENT, tag_all):
    """
    Returns the DataFrame of the Sensor Data from the database

    :param MEASUREMENT: table name
    :param tag_all: pid list
    :param client: connection to the database
    """
    tag = ", ".join("'" + str(a) + "'" for a in tag_all)
    sql_query = (
        'SELECT t_pid_no_text, date_trunc(\'hour\',msumt_datetime), msumt_value \nFROM "{mea}"\n'
        + "WHERE t_pid_no_text in ({pid})").format(
        **{
            "mea": MEASUREMENT,
            "pid": tag
        }
    )
    print(sql_query)
    try:
        cursor = client.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
        cursor.close()
        print("data fetched from db...")
    except Exception as e:
        print("in 2 try Oops!", e.__class__, "occurred.")
        with open('db_operations/t_iot_data.sql') as sql_file:
            cursor = client.cursor()
            sql_as_string = sql_file.read()
            cursor.execute(sql_as_string)
            cursor.execute("COMMIT;")
            cursor.close()

        cursor = client.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
        cursor.close()

    res_df = pd.DataFrame(result, columns=["pid_no", "time", "last"])
    res_df["time"] = pd.to_datetime(res_df["time"], format="%Y-%m-%dT%H:%M:%S.%f", utc=True)
    res_df["time"] = res_df["time"].dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    res_df = pd.pivot_table(res_df, index="time", columns=res_df.pid_no, values="last")

    return res_df


def get_threshold_value(conn, asset_id):
    """
    Get the threshold value for the asset_id from the database.
    If the unit_id is 3, then the threshold value is 51 else 65.
    """
    create_threshold_file = "db_operations/create_threshold_value.sql"

    cursor = conn.cursor()
    with open(create_threshold_file) as sql_file:
      sql_as_string = sql_file.read()
      cursor.execute(sql_as_string)
      conn.commit()

    select_threshold_df = pd.read_sql_query("SELECT * FROM t_threshold_value", conn)

    if len(select_threshold_df) == 0:
        """upload csv to t_threshold_value table in database if table is empty"""

        print("uploading csv to t_threshold_value table in database")

        relation_df = pd.read_csv('data/threshold_value.csv')
        for row in relation_df.itertuples():
            cursor.execute('''INSERT INTO t_threshold_value (unit, threshold)
                              VALUES ({}, {})'''.format(row.unit, row.threshold))

        conn.commit()


    '''Find the unit_id for the asset_id and then find the threshold value for that unit_id from the database.'''

    sql_query_unit = """SELECT unit_id FROM t_model_setsubi_training_data_check
                        WHERE asset_id = {} LIMIT 1""".format(asset_id)
    cursor.execute(sql_query_unit)
    unit_name = cursor.fetchall()
    unit_name = unit_name[0][0]

    sql_query_threshold = """SELECT threshold FROM t_threshold_value
                             WHERE unit = {} LIMIT 1""".format(int(unit_name))
    cursor.execute(sql_query_threshold)
    threshold_wear = cursor.fetchall()[0][0]

    return threshold_wear


def input_df_wear(one_asset_id, conn):
    """
    Returns the input data for the wear model

    :param one_asset_id: asset-id
    :param conn: connection to the database
    """

    cursor = conn.cursor()
    sql_query_raw = """SELECT measurement_record_set_id
                       FROM t_model_setsubi_raw_data
                       WHERE asset_id = {} AND is_deleted = False;""".format(one_asset_id)

    cursor.execute(sql_query_raw)
    measurement_record_set_id = cursor.fetchall()

    item_id_relation = """SELECT * FROM t_measurement_item_id_relation
                          WHERE asset_id ={}""".format(one_asset_id)

    relation = pd.read_sql_query(item_id_relation, conn)
    record_set_id = tuple([i[0] for i in measurement_record_set_id])

    print("record_set_id :", record_set_id)

    if not record_set_id:
        return

    sql_query_input = """SELECT * FROM t_model_setsubi_data_input
                         WHERE measurement_record_set_id in {} AND is_deleted = False;""".format(record_set_id)
    df_input = pd.read_sql_query(sql_query_input, conn)

    relation_measurement_item_id = relation['measurement_item_id'].to_list()
    relation_measurement_item_id.insert(0, 0)

    print("relation_measurement_item_id", relation_measurement_item_id)

    df_loop = df_input[df_input['measurement_item_id'].isin(relation_measurement_item_id)]

    list_for_df = []
    for i, j in df_loop.groupby(df_loop['measurement_record_set_id']):
        date_l = j[j['model_setsubi_column_master_id'].isin([1])].value.to_list()[0]
        hour_l = j[j['model_setsubi_column_master_id'].isin([2])].value.to_list()[0]
        df_l = j[j['model_setsubi_column_master_id'].isin([3])]
        for index in df_l.index:
            list_for_df.append((date_l, hour_l, df_l.loc[index, 'value'], df_l.loc[index, 'measurement_item_id']))

    df_use = pd.DataFrame(list_for_df,columns = ['date','hours', 'wear', 'measurement_item_id' ])
    df_use['date'] = pd.to_datetime(df_use['date'])
    convert_dict = {'hours': int, 'wear': float}

    df_use = df_use.astype(convert_dict)
    df_use['date'] = pd.to_datetime(df_use['date'] + timedelta(hours=9.5))
    df_use['date'] = pd.to_datetime(df_use["date"].dt.strftime('%Y-%m-%d'))
    df_use['date'] = pd.to_datetime(df_use["date"])

    df_wear_l = []
    for idx, df_g in df_use.groupby(['measurement_item_id']):
        df_diff = df_g.sort_values(by=['date'])
        df_diff.loc[:, "mill_hours_diff"] = df_diff["hours"].diff().values
        # df_diff.loc[:, "wear_diff"] = df_diff["wear"].diff().values
        df_diff.loc[:, "wear_diff"] = df_diff["wear"].diff().apply(lambda x: abs(x) if x <= 0 else x)
        df_diff.loc[:, "date_start"] = df_diff["date"].shift(1)
        df_diff.dropna(subset=["date_start", "wear_diff"], inplace=True)
        df_wear_l.append(df_diff)

    print("df_wear list created", len(df_wear_l))
    df_wear = pd.concat(df_wear_l, ignore_index=True)
    df_wear.insert(loc = 1, column = 'asset_id', value = one_asset_id)

    return df_wear, relation


def df_wear_cleaning(df_wear):
    """
    If wear is zero then delete the row.

    Replacement and maintainance record wear values must not get differenciated from previous record so
    the below line of code will prevent it
    """
    df_wear.loc[df_wear['mill_hours_diff']<0, 'wear_diff'] = df_wear.loc[df_wear['mill_hours_diff']<0, 'wear']
    df_wear.loc[df_wear['mill_hours_diff']<0, 'mill_hours_diff'] = df_wear.loc[df_wear['mill_hours_diff']<0, 'hours']
    #df_wear.drop(df_wear[df_wear['wear'] == 0].index, inplace=True)
    df_wear.drop(df_wear[df_wear['mill_hours_diff'] == 0].index , inplace=True)

    return df_wear

def calc_rate(df):
    """
    Create a column named diff_date_h which is the difference between the date and date_start in hours
    and a column named rate_mill_h which is the wear_diff divided by diff_date_h
    """

    df.loc[:, "diff_date_h"] = (df["date"] - df["date_start"]) / pd.Timedelta(1, "h")
    df.loc[:, "rate_mill_h"] = df.mill_hours_diff / df.diff_date_h

    return df

def all_asset_pid(asset_id, conn):
    """Get all the pids for the Asset ID"""
    sql_query_pid = """SELECT * FROM t_mill_model_pid_relation
                       WHERE asset_id = {};""".format(asset_id)
    df_asset_pids = pd.read_sql_query(sql_query_pid, conn)

    return df_asset_pids

def columns_dot_product(A, B):
    return inner1d(A.T, B.T)


def supply_current_convert(N, mill, t, state_minmax, state_minmax_notna, state_std, state_std_notna, V_std):
    term_supply =[]
    term_current=[]
    N_tp = []
    supply = []
    term_supply_na = np.full((N, V_std-2),1)
    term_current_na = np.full((N, 2),1)
    for i in range(N):
        N_tp.append(sum(state_minmax_notna[mill,t[i, 0]:t[i, 1]+1, 0]))
        supply.append(sum(state_minmax[mill,t[i, 0]:t[i, 1]+1, 0]))

        term_supply.append(columns_dot_product(state_std[mill,t[i, 0]:t[i, 1]+1, :(V_std - 2)],
                                    state_std_notna[mill,t[i, 0]:t[i, 1]+1, :(V_std - 2)]))
        for v in range(0,(V_std - 2)):
            T_valid = sum(state_std_notna[mill,t[i, 0]:t[i, 1]+1, v])
            if (T_valid > 0):
                term_supply[i][v] = term_supply[i][v] /T_valid
            else:
                term_supply_na[i][v] = 0

        term_current.append(columns_dot_product(state_std[mill,t[i, 0]:t[i, 1]+1, (V_std - 2):],
                                    state_std_notna[mill,t[i, 0]:t[i, 1]+1, (V_std - 2):]))
        for v in range(0, 2):
            T_valid = sum(state_std_notna[mill,t[i, 0]:t[i, 1]+1, V_std - 2 + v])
            if (T_valid > 0):
                term_current[i][v] = term_current[i][v] /T_valid
            else:
                term_current_na[i][v] = 0

    term_supply =np.array(term_supply)
    term_current=np.array(term_current)
    N_tp = np.array(N_tp)
    supply = np.array(supply)
    term_supply_na = np.array(term_supply_na)
    term_current_na = np.array(term_current_na)

    return term_supply, term_current, N_tp, supply, term_supply_na, term_current_na

def scale_mill(vnames, df_ai, scaler, test_start):
    state = np.empty((1, len(df_ai), len(vnames)))
    for i,v in enumerate(vnames):
        all_mill = []
        for j, m in enumerate(v):
            state[j, :, i] = df_ai[m].values
            # スケールの基準はtrainのみ
            all_mill.append(df_ai.loc[:test_start, m].values)
        all_mill = np.concatenate(all_mill).reshape(-1, 1)
        scaler.fit(all_mill)
        for m in range(1):
            state[m, :, i] = scaler.transform(state[m, :, [i]].reshape(-1, 1)).ravel()
        state_notna = (~np.isnan(state)).astype(float)
        #         mean_train = np.nanmean(all_mill)
        #         np.nan_to_num(state[:, :, i], copy=False, nan=mean_train)
        np.nan_to_num(state[:, :, i], copy=False, nan=0)
    return state, state_notna

def scale_mill_state(vnames, df_ai, scaler):

    """Need to conferm thre was to function with same name (scale_mill)"""
    state = np.empty((1, len(df_ai), len(vnames)))
    for i, v in enumerate(vnames):
        #         all_mill = []
        for j, m in enumerate(v):
            state[j, :, i] = df_ai[m].values
    return state

def estimate_operation_rate(wear, supply_test, N_tp_test, supply_rate):
    """Estimate the current operation rate based on the wear data and the supply rate data."""
    mean_operation = np.dot(wear[:,1], wear[:,2]) / sum(wear[:,2])
    mean_supply_rate = np.dot(supply_rate, wear[:,2]) / sum(wear[:,2])
    slope = sum((wear[:,1] - mean_operation) * (supply_rate - mean_supply_rate) * wear[:,2]) /sum(np.square(wear[:,1] - mean_operation) * wear[:,2])
    intercept = (mean_supply_rate - (slope * mean_operation))

    est_current_operation_rate = ((supply_test / N_tp_test) - intercept) / slope

    return est_current_operation_rate

def getCurrentSensorCount(conn,asset_id,highest_date):
    sensorCountSql = "db_operations/sensorCount.sql"
    current_date = str(datetime.now())
    cursor = conn.cursor()

    with open(sensorCountSql) as sql_file:
      sql_as_string = sql_file.read()
      cursor.execute(sql_as_string.replace("{asset_id}", str(asset_id)).replace("{highest_date}", str(highest_date)).replace("{current_date}", current_date))
      total_sensor_count = cursor.fetchall()
      total_sensor_count = total_sensor_count[0][0]

    print("total_sensor_count :", total_sensor_count)
    return total_sensor_count

# Train Functions

def msumt_item_relation_table(env_value, conn):
    """
    Create a table named t_measurement_item_id_relation in the database if it does not exist
    and upload the csv file to the table. If the table already exists, the csv file is not uploaded.

    :param env_value: 'Dev' or 'Stg' or 'Prod'
    :param conn: connection to the database
    """
    create_msumt_item_relation = "db_operations/create_t_msumt_item_relation.sql"

    cursor = conn.cursor()
    with open(create_msumt_item_relation) as sql_file:
      sql_as_string = sql_file.read()
      cursor.execute(sql_as_string)
      conn.commit()

    select_relation = """SELECT * FROM t_measurement_item_id_relation;"""

    select_relation_df = pd.read_sql_query(select_relation, conn)

    if len(select_relation_df) == 0:
        # upload csv to t_measurement_item_id_relation table in database if table is empty
        print("uploading csv to t_measurement_item_id_relation table in database")
        relation_df = pd.read_csv('data/{}_test_t_measurement_item_id_relation.csv'.format(env_value))
        for row in relation_df.itertuples():
            cursor.execute('''
                            INSERT INTO t_measurement_item_id_relation (asset_id, unit, mill, roller, longitude, measurement_item_id)
                            VALUES ({}, {}, \'{}\', {}, {}, {})
                            '''.format(row.asset_id, row.unit, row.mill, row.roller_no, row.longitude, row.measurement_item_id)
                            )

        conn.commit()

def model_pid_relation(conn):
    """
    Create Table If not Exists and Upload CSV to Table in Database.

    :param conn: connection to the database
    """
    create_pid_relation = "db_operations/create_model_pid_relation.sql"

    cursor = conn.cursor()
    with open(create_pid_relation) as sql_file:
      sql_as_string = sql_file.read()
      cursor.execute(sql_as_string)
      conn.commit()

    select_relation = """SELECT * FROM t_mill_model_pid_relation;"""

    select_relation_df = pd.read_sql_query(select_relation, conn)

    if len(select_relation_df) == 0:
        """upload csv to t_mill_model_pid_relation table in database if table is empty"""

        print("uploading csv to t_mill_model_pid_relation table in database")
        relation_df = pd.read_csv('data/t_mill_model_pid_relation.csv')
        for row in relation_df.itertuples():
            cursor.execute('''INSERT INTO t_mill_model_pid_relation (t_pid_no_text, asset_id, sensor_name)
                              VALUES (\'{}\', {}, \'{}\')'''.format(row.t_pid_no_text, row.asset_id, row.sensor_name))

        conn.commit()

def sim_days_calculate(wear_threshold, total_wear, model_calc_rate, rate_mill_hours, utilization_val_mean):
    
    initial_remaining_days = (wear_threshold - total_wear) / model_calc_rate
    
    if initial_remaining_days > 2190:
        initial_remaining_days = 2190
    
    # if initial_remaining_days < -5475:
    #     initial_remaining_days = -5475

    utilization_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    remaining_days = []
    for rate in utilization_rates:
        adjusted_rate_mill_hours = rate * rate_mill_hours
        adjusted_remaining_days = initial_remaining_days * ((rate_mill_hours / adjusted_rate_mill_hours) - utilization_val_mean*rate_mill_hours)
        
        remaining_days.append(adjusted_remaining_days)
    
    utilization_val_mean_exact = int(str(int(utilization_val_mean*100))[0])/10
    predicted_index = np.array([round(i, 3) for i in np.abs(np.array(utilization_rates) - utilization_val_mean_exact)]).argmin()
    
    denominator = remaining_days[predicted_index]
    numerator = initial_remaining_days
    
    #temp = int(str(int(utilization_val_mean*100))[1])/100
    temp = utilization_val_mean - utilization_val_mean_exact
    
    propotion = 1-(numerator/denominator)-temp
    
       
    for i in range(len(remaining_days)):
        remaining_days[i] = remaining_days[i]-(remaining_days[i]*propotion)
    print("remaining_days: ")
    print(remaining_days)
    return remaining_days

#For Manual Prediction 
def db_to_pandas_manual(client, MEASUREMENT, tag_all, unit):
    tag = ",".join("'" + str("HE_A{}00=".format(unit)+a) + "'" for a in tag_all)
    sql_query = (
        'SELECT t_pid_no_text, date_trunc(\'hour\',msumt_datetime), msumt_value \nFROM "{mea}"\n'
        + "WHERE t_pid_no_text in ({pid})"
    ).format(
        **{
            "mea": MEASUREMENT,
            "pid": tag,
        }
    )
    print(sql_query)
    try:
        cursor = client.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
        cursor.close()
    except Exception as e:
        print(e)
        with open('tables.sql') as sql_file:
            cursor = client.cursor()
            sql_as_string = sql_file.read()
            cursor.execute(sql_as_string)
            cursor.execute("COMMIT;")
            cursor.close()

        cursor = client.cursor()
        cursor.execute(sql_query)
        result = cursor.fetchall()
        cursor.close()

    res_df = pd.DataFrame(result, columns=["pid_no", "time", "last"])
    res_df["pid_no"].replace(to_replace="HE_A{}00=".format(unit), value='', regex=True, inplace=True)
    res_df["time"] = pd.to_datetime(res_df["time"], format="%Y-%m-%dT%H:%M:%S.%f", utc=True)
    res_df["time"] = res_df["time"].dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)

    res_df = pd.pivot_table(res_df, index="time", columns=res_df.pid_no, values="last")

    return res_df

# For Manual Prediction
def supply_current_convert_manual(N, mill, t, state_minmax, state_minmax_notna, state_std, state_std_notna, V_std):
    term_supply =[]
    term_current=[]
    N_tp = []
    supply = []
    term_supply_na = np.full((N, V_std-2),1)
    term_current_na = np.full((N, 2),1)
    for i in range(N):
        N_tp.append(sum(state_minmax_notna[mill[i]-1,t[i, 0]:t[i, 1]+1, 0]))
        supply.append(sum(state_minmax[mill[i]-1,t[i, 0]:t[i, 1]+1, 0]))

        term_supply.append(columns_dot_product(state_std[mill[i]-1,t[i, 0]:t[i, 1]+1, :(V_std - 2)],
                                    state_std_notna[mill[i]-1,t[i, 0]:t[i, 1]+1, :(V_std - 2)]))
        for v in range(0,(V_std - 2)):
            T_valid = sum(state_std_notna[mill[i]-1,t[i, 0]:t[i, 1]+1, v])
            if (T_valid > 0):
                term_supply[i][v] = term_supply[i][v] /T_valid
            else:
                term_supply_na[i][v] = 0

        term_current.append(columns_dot_product(state_std[mill[i]-1,t[i, 0]:t[i, 1]+1, (V_std - 2):],
                                    state_std_notna[mill[i]-1,t[i, 0]:t[i, 1]+1, (V_std - 2):]))
        for v in range(0, 2):
            T_valid = sum(state_std_notna[mill[i]-1,t[i, 0]:t[i, 1]+1, V_std - 2 + v])
            if (T_valid > 0):
                term_current[i][v] = term_current[i][v] /T_valid
            else:
                term_current_na[i][v] = 0

    term_supply =np.array(term_supply)
    term_current=np.array(term_current)
    N_tp = np.array(N_tp)
    supply = np.array(supply)
    term_supply_na = np.array(term_supply_na)
    term_current_na = np.array(term_current_na)

    return term_supply, term_current, N_tp, supply, term_supply_na, term_current_na

# For Manual Prediction
def scale_mill_manual(vnames, df_ai, scaler, test_start):
    state = np.empty((6, len(df_ai), len(vnames)))
    for i, v in enumerate(vnames):
        all_mill = []
        for j, m in enumerate(v):
            state[j, :, i] = df_ai[m].values
            # スケールの基準はtrainのみ
            all_mill.append(df_ai.loc[:test_start, m].values)
        all_mill = np.concatenate(all_mill).reshape(-1, 1)
        scaler.fit(all_mill)
        for m in range(6):
            state[m, :, i] = scaler.transform(state[m, :, [i]].reshape(-1, 1)).ravel()
        state_notna = (~np.isnan(state)).astype(float)
        #         mean_train = np.nanmean(all_mill)
        #         np.nan_to_num(state[:, :, i], copy=False, nan=mean_train)
        np.nan_to_num(state[:, :, i], copy=False, nan=0)

    return state, state_notna

# For Manual Prediction
def scale_mill_state_manual(vnames, df_ai, scaler):
    state = np.empty((6, len(df_ai), len(vnames)))
    for i, v in enumerate(vnames):
        #         all_mill = []
        for j, m in enumerate(v):
            state[j, :, i] = df_ai[m].values

    return state

def append_earliest(df):
    df_early = []
    for i, df_i in df.groupby(["mill", "roller"]):
        df_append = df_i.loc[
            df_i["threshold_date"] == df_i["threshold_date"].min(),
            ["mill", "roller", "rate_mill_h", "threshold_date"],
        ].iloc[[0], :]
        df_append.loc[:, "longitudinal"] = "最短"
        df_append = df_append.append(
            df_i[
                ["mill", "roller", "longitudinal", "rate_mill_h", "threshold_date"]
            ].copy()
        )
        df_early.append(df_append)


    for i, df_i in df.groupby("mill"):
        df_append = df_i.loc[
            df_i["threshold_date"] == df_i["threshold_date"].min(),
            ["mill", "roller", "rate_mill_h", "threshold_date"],
        ].iloc[[0], :]
        df_append.loc[:, "roller"] = "最短"
        df_append.loc[:, "longitudinal"] = "最短"
        df_early.append(df_append)

    df_early = pd.concat(df_early)

    return df_early

def append_earliest_sim(df):
    df_early = []
    for i, df_i in df.groupby(["mill", "roller", "rate_mill_h"]):
        df_append = df_i.loc[
            df_i["threshold_date"] == df_i["threshold_date"].min(),
            ["mill", "roller", "rate_mill_h", "threshold_date"],
        ].iloc[[0], :]
        df_append.loc[:, "longitudinal"] = "最短"
        df_early.append(df_append)
        df_early.append(
            df_i[["mill", "roller", "rate_mill_h", "threshold_date", "longitudinal"]]
        )

    for i, df_i in df.groupby(["mill", "rate_mill_h"]):
        df_append = df_i.loc[
            df_i["threshold_date"] == df_i["threshold_date"].min(),
            ["mill", "roller", "rate_mill_h", "threshold_date"],
        ].iloc[[0], :]
        df_append.loc[:, "roller"] = "最短"
        df_append.loc[:, "longitudinal"] = "最短"
        df_early.append(df_append)

    df_early = pd.concat(df_early)

    return df_early