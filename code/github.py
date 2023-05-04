import math
from pyspark import SparkContext
import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb

def get_stat_dict(name, number):
    stats = {
        f"{name}_avg": number.mean(),
        f"{name}_std": number.std(),
        f"{name}_kurt": number.kurt(),
        f"{name}_skew": number.skew(),
        f"{name}_max": number.max(),
        f"{name}_min": number.min(),
    }   
    return stats

def parse_attribute_value(value):
    if isinstance(value, dict):
        return sum([1 if v in ('True', 'true') else 0 if v in ('False', 'false') else float(v) if v else np.nan for v in value.values()])
    elif isinstance(value, bool):
        return 1 if value else 0
    elif isinstance(value, str):
        return 1 if value in ('True', 'true') else 0 if value in ('False', 'false') else float(value) if value else np.nan
    else:
        return float(value) if value else np.nan

def extract_feature_info(feature_vec, entity_type):
    # """
    # :param feature:The vector we want to get its feature collection
    # :param type: The type of estimation we need
    # :return: the feature output
    # """
    # rst = dict()
    # if len(feature) <= 0:
    #     return user_default if type == "user" else business_default
    # array_list = pd.Series([float(i[1]) for i in feature])
    # rst[f"{type}_avg"] = array_list.mean()
    # rst[f"{type}_std"] = array_list.std()
    # rst[f"{type}_kurt"] = array_list.kurt()
    # rst[f"{type}_skew"] = array_list.skew()
    # rst[f"{type}_max"] = array_list.max()
    # rst[f"{type}_min"] = array_list.min()
    # return rst

    if entity_type == "user":
        default_stats = user_default 
    else:
        default_stats = business_default

    if len(feature_vec) == 0:
        return default_stats
    
    # Extract features
    values = pd.Series([float(i[1]) for i in feature_vec])
    return get_stat_dict(entity_type, values)


def update_missing_variables(data):
    data['business_avg'].fillna(business_default['business_avg'],inplace=True)
    data['business_std'].fillna(business_default['business_std'], inplace=True)
    data['business_kurt'].fillna(business_default['business_kurt'], inplace=True)
    data['business_skew'].fillna(business_default['business_skew'], inplace=True)
    data['business_max'].fillna(business_default['business_avg'], inplace=True)
    data['business_min'].fillna(business_default['business_avg'], inplace=True)
    data.fillna(0, inplace=True)
    # time data
    data["history"] = (2023 - data["yelping_since"].transform(lambda x: int(x[:4]))) * 12 + data["yelping_since"].transform(lambda x: int(x[5:7]))
    
    return data

def vector_column_generation(user_id, business_id):
    base_dict = {"user_id": user_id, "business_id": business_id}

    user_vector = user_dict.get(user_id, [])
    user_vector = [i for i in user_vector if i[0] != business_id]
    user_vector_dict = extract_feature_info(user_vector, "user")

    business_vector = business_dict.get(business_id, [])
    business_vector = [i for i in business_vector if i[0] != user_id]
    business_vector_dict = extract_feature_info(business_vector, "business")

    combined_vector_dict = dict(user_vector_dict, **business_vector_dict)
    return dict(base_dict, **combined_vector_dict)

def combine_statistics(user,business_id,score,user_list,business_list):
    try:
        u_raw_list = user_list.remove(score)
        bu_raw_list = business_list.remove(score)
    except:
        pass
    user_stat_dic = get_statistics(u_raw_list, temp=user_default, label='user')    
    bu_stat_dic = get_statistics(bu_raw_list, temp=business_default, label='business')
    # initial = {'user_id':user,'business_id':business_id,'y':score}
    return {'user_id':user,'business_id':business_id,'y':score,**user_stat_dic,**bu_stat_dic}

def get_statistics(raw_list,temp=None,label='user'):
    np_list = pd.Series(raw_list)
    # stat_dict = dict()
    # if raw_list and len(raw_list) > 0:
    #     stat_dict[f'{label}_avg'] = np_list.mean()
    #     stat_dict[f'{label}_std'] = np_list.std()
    #     stat_dict[f'{label}_kurt'] = np_list.kurt()
    #     stat_dict[f'{label}_skew'] = np_list.skew()
    #     stat_dict[f'{label}_max'] = np_list.max()
    #     stat_dict[f'{label}_min'] = np_list.min()
    #     return stat_dict
    # else:
    #     return temp

    if raw_list and len(raw_list) > 0:
        # stats = {
        #     f"{label}_avg": np_list.mean(),
        #     f"{label}_std": np_list.std(),
        #     f"{label}_kurt": np_list.kurt(),
        #     f"{label}_skew": np_list.skew(),
        #     f"{label}_max": np_list.max(),
        #     f"{label}_min": np_list.min(),
        # } 
        return get_stat_dict(label, np_list)
    else:
        return temp

def merge_dataframes(data, feature_list, users):
    data = pd.DataFrame(data)
    user_data = pd.DataFrame(users)

    for features in feature_list:
        temp_data = pd.DataFrame(features)
        data = pd.merge(data, temp_data, on = "business_id", how = "left")

    data = pd.merge(data, user_data, on="user_id", how="left")

    return data

def createXandY(train_data):
    train_data["y"] = train_data["y"].astype("float")
    train_cols = train_data.columns.difference(["y", "business_id", "user_id", "yelping_since"])

    return train_data[train_cols], train_data["y"], train_cols

if __name__ == "__main__":
    sc = SparkContext()
    sc.setLogLevel("ERROR")

    # folder_path = sys.argv[1]
    # input_test_file = sys.argv[2]
    # output_file = sys.argv[3]

    folder_path = "data"
    input_test_file = "data/yelp_val.csv"
    output_file = "result/task.csv"

    # Other input paths

    user_file = "/user.json"
    review_file = "/review_train.json"
    business_file = "/business.json"
    checkin_file = "/checkin.json"
    tip_file = "/tip.json"
    photo_file = "/photo.json"

    # Variables
    NUM_OF_PARTITIONS = 20
    REPARTITIONS = 50
    LABELS = ['drink', 'food', 'inside', 'menu', 'outside']
    UNIQUE_VALUE_THRESHOLD = 10
    MINIMUM_BUSINESS_THRESHOLD = 0.2
    business_default = {"business_avg": 3.75088,"business_std": 0.990780,"business_kurt": 0.48054,"business_skew": -0.70888,'business_max': 5, 'business_min': 1}
    user_default = {"user_avg": 3.75117,"user_std": 1.03238,"user_kurt": 0.33442,"user_skew": -0.70884,'user_max':5,'user_min':1}

    # Get all RDDs from data

    train_data_RDD = sc.textFile(folder_path + "/yelp_train.csv")
    header = train_data_RDD.first()
    train_data_RDD = train_data_RDD.filter(lambda row: row != header).map(lambda row: row.split(","))

    test_data_RDD = sc.textFile(input_test_file)
    header = test_data_RDD.first()
    test_data_RDD = test_data_RDD.filter(lambda row: row != header).map(lambda row: row.split(","))

    user_agg_rdd = train_data_RDD.map(lambda x:(x[0],float(x[2]))).groupByKey()
    bu_agg_rdd = train_data_RDD.map(lambda x:(x[1],float(x[2]))).groupByKey()

    user_dict = train_data_RDD.map(lambda x: (x[0], (x[1], x[2]))).map(lambda x: (x[0], x[1])).groupByKey().mapValues(lambda x: list(x)).collectAsMap()
    business_dict = train_data_RDD.map(lambda x: (x[1], (x[0], x[2]))).map(lambda x: (x[0], x[1])).groupByKey().mapValues(lambda x: list(x)).collectAsMap()

    train_rdd = train_data_RDD.map(lambda x:(x[0],(x[1],float(x[2]))))\
        .join(user_agg_rdd)\
        .map(lambda x:(x[1][0][0],(x[0],x[1][0][1],x[1][1])))\
        .join(bu_agg_rdd)\
        .map(lambda x:(x[1][0][0],x[0],x[1][0][1],x[1][0][2],x[1][1]))\
        .repartition(REPARTITIONS)\
        .map(lambda x:combine_statistics(x[0],x[1],x[2],list(x[3]),list(x[4])))\
        .collect()


    # Get features

    # from user.json
    user_test_data_RDD = test_data_RDD.map(lambda row: (row[0], row[1]))
    test_user_index = set(user_test_data_RDD.map(lambda x: x[0]).distinct().collect() + list(user_dict.keys()))
    lines_user = sc.textFile(folder_path + user_file).map(lambda row: json.loads(row))
    users = lines_user.filter(lambda x: x["user_id"] in test_user_index).collect()
    for user in users:
        del user["name"]
        user["friends"] = 0 if user["friends"] == "None" else len(user["friends"].split(","))
        user["elite"] = 0 if user["elite"] == "None" else len(user["elite"].split(","))

    # from tips.json
    lines_tips = sc.textFile(folder_path + tip_file).map(lambda x: json.loads(x))
    tips = lines_tips.map(lambda x: (x['business_id'], (x['likes'], 1))) \
        .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))\
        .map(lambda x: {"business_id": x[0], "likes_sum": x[1][0], "likes_count": x[1][1]})\
        .collect()
    
    # from checkin.json
    lines_checks = sc.textFile(folder_path + checkin_file).map(lambda x: json.loads(x))
    checks = lines_checks.map(lambda x: (x["business_id"], (sum(list(x["time"].values())), len(list(x["time"].values())))))\
        .map(lambda x:{'business_id':x[0],'slots':x[1][0],'customers':x[1][1]})\
        .collect()

    # from photo.json 
    lines_photo = sc.textFile(folder_path + photo_file).map(lambda row: json.loads(row))
    photo_dict = lines_photo.map(lambda row: (row['business_id'], [row["label"]])).reduceByKey(lambda x, y: x + y).collect()

    photos = list()

    for row in photo_dict:
        rcd = {label: row[1].count(label) for label in LABELS}
        rcd["photo_sum"] = sum(rcd.values())
        rcd["business_id"] = row[0]
        photos.append(rcd)

    # from business.json
    lines_business =  sc.textFile(folder_path + business_file).map(lambda x: json.loads(x))
    business = lines_business.map(lambda x: (x["business_id"], (x["stars"], x["review_count"], x['latitude'], x['longitude'], x['is_open'])))\
        .map(lambda bu:{"business_id": bu[0], "b_stars": bu[1][0], "b_review_count": bu[1][1], 'latitude': bu[1][2],
                 'longitude': bu[1][3], 'is_open': bu[1][4]})\
        .collect()

    num_business = lines_business.count()

    lines_business = lines_business.collect()
    attribute_dict = dict()
    attribute_count = dict()
    
    for row in lines_business:
        if row.get('attributes',None):
            sub_dic = row['attributes']
            for j in sub_dic.keys():
                attribute_dict.setdefault(j, set())
                attribute_dict[j].add(sub_dic[j])
                attribute_count.setdefault(j, 0)
                attribute_count[j] += 1 / num_business  

    selected_attributes = [key for key,value in attribute_dict.items() if len(value) <= UNIQUE_VALUE_THRESHOLD and attribute_count[key] >= MINIMUM_BUSINESS_THRESHOLD]
    
    temp_dic = {i: None for i in selected_attributes}

    business_attribute = list()

    for row in lines_business:
        bus_record = temp_dic.copy()
        bus_record["business_id"] = row["business_id"]
        if row.get('attributes',None):
            sub_dic = row['attributes']
            for j in sub_dic.keys():
                if j in selected_attributes:
                    bus_record[j] = sub_dic[j]
            business_attribute.append(bus_record.copy())

    
    # 3. merge all of dataframe to get the whole data

    all_features = [business, business_attribute, tips, photos, checks]

    train_data = merge_dataframes(train_rdd, all_features, users)

    # Create one-hot encoding
    business_attributes_df = pd.DataFrame(business_attribute)

    for attribute in business_attributes_df.columns:
        if attribute == 'business_id':
            continue
        else:
            # Create dummy variables for the attribute
            attribute_values = train_data[attribute].fillna('Unknown')
            dummy_variables = pd.get_dummies(attribute_values, drop_first=True)
            
            # Rename the columns of the dummy variables with the attribute name
            new_cols = [attribute + i for i in dummy_variables.columns]
            dummy_variables.columns = new_cols
            
            # Remove the original attribute column from the dataset and concatenate the dummy variables
            train_data = pd.concat([train_data.drop(attribute, axis=1), dummy_variables], axis=1)
    
    train_data = update_missing_variables(train_data)

    # Do the same thing to the test data
    test_data = test_data_RDD.repartition(NUM_OF_PARTITIONS).map(lambda x: vector_column_generation(x[0], x[1])).collect()
    test_data = merge_dataframes(test_data, all_features, users)

    valid_attributes = train_data.columns

    for attribute in business_attributes_df.columns:
        if attribute == 'business_id':
            continue
        else:
            # Create dummy variables for the attribute
            attribute_values = test_data[attribute].fillna('Unknown')
            dummy_variables = pd.get_dummies(attribute_values, drop_first=True)

            # Get the columns that exist in both the dummy variables and the train data
            cols = [i for i in dummy_variables.columns if attribute + i in valid_attributes]
            dummy_variables = dummy_variables[cols]
            
            # Rename the columns of the dummy variables with the attribute name
            new_cols = [attribute + i for i in dummy_variables.columns]
            dummy_variables.columns = new_cols
            
            # Remove the original attribute column from the dataset and concatenate the dummy variables
            test_data = pd.concat([test_data.drop(attribute, axis=1), dummy_variables], axis=1)
    
    test_data = update_missing_variables(test_data)

  # 4. XGboost
    train_x, train_y, train_cols = createXandY(train_data)

    # model = xgb.XGBRegressor(
    #     max_depth=5,
    #     min_child_weight=1,
    #     subsample=0.6,
    #     colsample_bytree=0.6,
    #     gamma=0,
    #     reg_alpha=1,
    #     reg_lambda=0,
    #     learning_rate=0.05,
    #     n_estimators=800
    # )

    param = {
        'lambda': 9.92724463758443, 
        'alpha': 0.2765119705933928, 
        'colsample_bytree': 0.5, 
        'subsample': 0.8, 
        'learning_rate': 0.02, 
        'max_depth': 17, 
        'random_state': 2020, 
        'min_child_weight': 101,
        'n_estimators': 300,
    }
    model = xgb.XGBRegressor(**param)


    model.fit(train_x, train_y, early_stopping_rounds=20, eval_set=[(train_x, train_y)], verbose=50)

    prediction_df = pd.DataFrame(model.predict(test_data[train_cols]))

    preds = pd.concat([test_data[["user_id", "business_id"]], prediction_df], axis=1)

    preds.columns = ["user_id", "business_id", "prediction"]

    preds.to_csv(output_file, index=False)

    # validation
    val_data = pd.read_csv(input_test_file)
    valid_set = val_data.merge(preds,on=['user_id','business_id'])
    valY = np.array(valid_set['stars'])
    prd_array = np.array(valid_set['prediction'])

    def RMSE(np1,np2):
        return math.sqrt(sum((np1-np2)**2)/len(np1))

    print(RMSE(valY,prd_array))