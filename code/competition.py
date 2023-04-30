import math
from pyspark import SparkContext
import sys
import json
import numpy as np
import xgboost as xgb

def write_csv_file(result_str, output_file):
    with open(output_file, "w") as f:
        result_header = "user_id, business_id, prediction\n"
        f.writelines(result_header)
        f.writelines(result_str)

def createXandY(data_RDD):
    X = list()
    Y = list()
    for user, bus, rating in data_RDD.collect():
        Y.append(rating)
        review = review_dict.get(bus, (None, None, None))
        user_info = user_dict.get(user, (None, None, None))
        bus_info = business_dict.get(bus, (None, None))
        X.append([review[0], review[1], review[2], user_info[0], user_info[1], user_info[2], bus_info[0], bus_info[1]])

    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')
    return X, Y

def createXtest(data_RDD):
    X = list()
    user_bus_list = list()
    for user, bus, stars in data_RDD.collect():
        user_bus_list.append((user, bus))
        review = review_dict.get(bus, (None, None, None))
        user_info = user_dict.get(user, (None, None, None))
        bus_info = business_dict.get(bus, (None, None))
        X.append([review[0], review[1], review[2], user_info[0], user_info[1], user_info[2], bus_info[0], bus_info[1]])

    X = np.array(X, dtype='float32')
    return X, user_bus_list

if __name__ == "__main__":
    sc = SparkContext()
    sc.setLogLevel("ERROR")

    folder_path = sys.argv[1]
    input_test_file = sys.argv[2]
    output_file = sys.argv[3]

    # folder_path = "data"
    # input_test_file = "data/yelp_val.csv"
    # output_file = "result/task.csv"

    #other input paths

    user_file = "/user.json"
    review_file = "/review_train.json"
    business_file = "/business.json"
    checkin_file = "/checkin.json"

    train_data_RDD = sc.textFile(folder_path + "/yelp_train.csv")
    header = train_data_RDD.first()
    train_data_RDD = train_data_RDD.filter(lambda row: row != header).map(lambda row: row.split(","))

    test_data_RDD = sc.textFile(input_test_file)
    header = test_data_RDD.first()
    test_data_RDD = test_data_RDD.filter(lambda row: row != header).map(lambda row: row.split(","))

    business_user_dict = train_data_RDD.map(lambda row: (row[0], row[1])).groupByKey().mapValues(set).collectAsMap()
    user_business_dict = train_data_RDD.map(lambda row: (row[1], row[0])).groupByKey().mapValues(set).collectAsMap()

    ## MODEL-BASED ##

    #collect features

    # from review.json

    lines_review = sc.textFile(folder_path + review_file).map(lambda row: json.loads(row))
    review_dict = lines_review.map(lambda row: (row['business_id'], (float(row['useful']), float(row['funny']), float(row['cool']), 1))) \
                        .reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1], x[2]+y[2], x[3]+y[3])) \
                        .mapValues(lambda x: (x[0]/x[3], x[1]/x[3], x[2]/x[3])) \
                        .collectAsMap()

    # from business.json

    lines_business = sc.textFile(folder_path + business_file).map(lambda row: json.loads(row))
    business_dict = lines_business.map(lambda row: (row['business_id'], (float(row['stars']), float(row['review_count'])))) \
                        .collectAsMap()

    # from user-001.json

    lines_user = sc.textFile(folder_path + user_file).map(lambda row: json.loads(row))
    user_dict = lines_user.map(lambda row: (row['user_id'], (float(row['average_stars']), float(row['review_count']), float(row['fans'])))) \
                        .collectAsMap()
    

    # make X-train and Y-train

    X_train, Y_train = createXandY(train_data_RDD)

    # make X-test and Y-test

    X_test, user_bus_list = createXtest(test_data_RDD)

    # xgbr = xgb.XGBRegressor(verbosity=0, n_estimators=30, random_state=1, max_depth=7)

    # xgbr = xgb.XGBRegressor(
    #     max_depth=7,
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
    xgbr = xgb.XGBRegressor(**param)

    xgbr.fit(X_train, Y_train)      

    model_based_result = xgbr.predict(X_test)


    ## HYBRID MODEL ##
    result_str = ""
    for i in range(len(model_based_result)):
        result_str += user_bus_list[i][0] + "," + user_bus_list[i][1] + "," + str(model_based_result[i]) + "\n"

    write_csv_file(result_str, output_file)

    # Calculate RMSE

    with open("result/task.csv") as in_file:
        guess = in_file.readlines()[1:]
    with open("data/yelp_val.csv") as in_file:
        ans = in_file.readlines()[1:]
    res = {"<1": 0, "1~2": 0, "2~3": 0, "3~4": 0, "4~5": 0}
    dist_guess = {"<1": 0, "1~2": 0, "2~3": 0, "3~4": 0, "4~5": 0}
    dist_ans = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    large_small = {"large": 0, "small": 0}

    RMSE = 0
    for i in range(len(guess)):
        diff = float(guess[i].split(",")[2]) - float(ans[i].split(",")[2])
        RMSE += diff**2
        if abs(diff) < 1:
            res["<1"] = res["<1"] + 1
        elif 2 > abs(diff) >= 1:
            res["1~2"] = res["1~2"] + 1
        elif 3 > abs(diff) >= 2:
            res["2~3"] = res["2~3"] + 1
        elif 4 > abs(diff) >= 3:
            res["3~4"] = res["3~4"] + 1
        else:
            res["4~5"] = res["4~5"] + 1
    RMSE = (RMSE/len(guess))**(1/2)
    print("RMSE: "+str(RMSE))
