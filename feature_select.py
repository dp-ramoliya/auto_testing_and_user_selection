import ast
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression

def feature_select_GB(X, y, asset_id, msumt_id):
    """
    Returns the List of Features

    :param X: total features
    :param y: target 
    :param asset_id: asset id
    :param msumt_id: measurement item id (logitude)
    """
    gb_model = GradientBoostingRegressor()

    gb_model.fit(X, y)
    gb_feature_importances = gb_model.feature_importances_
    gb_feature_importance_data = list(zip(X.columns, gb_feature_importances))
    sorted_gb_feature_importance = sorted(gb_feature_importance_data, key=lambda x: x[1], reverse=True)

    feature_l = []
    for feature, importance in sorted_gb_feature_importance:
        print(f"{feature}: {importance}")
        feature_l.append(feature)

    print("feature list: ", feature_l)
    user_sensor_index = input("Enter sensor index From above list to select Feature :")
    user_sensor_index_list = [int(i) for i in user_sensor_index]
    user_sensor_list = [feature_l[i] for i in user_sensor_index_list]
    return user_sensor_list

def feature_select_regression(X, y):

    num_features_to_select = 7  
    selector = SelectKBest(score_func=f_regression, k=num_features_to_select)
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_indices]

    feature_scores = selector.scores_

    feature_scores_dict = {feature: score for feature, score in zip(X.columns, feature_scores)}

    sorted_feature_scores = sorted(feature_scores_dict.items(), key=lambda x: x[1], reverse=True)

    feature_l = []
    for feature, score in sorted_feature_scores:
        if feature in selected_features:
            print(f"{feature}: {score}")
            feature_l.append(feature)
    print("feature list: ", feature_l)
    # user_sensor_list = ast.literal_eval(input("Enter sensor list From above list:"))
    user_sensor_index = input("Enter sensor index From above list to select Feature :")
    user_sensor_index_list = [int(i) for i in user_sensor_index]
    user_sensor_list = [feature_l[i] for i in user_sensor_index_list]
    return user_sensor_list
