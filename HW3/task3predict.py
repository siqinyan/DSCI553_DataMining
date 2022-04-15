#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
    Task3: Collaborative Filtering Recommendation System (5pts)

    Task description
    In this task, you will build collaborative filtering (CF) recommendation systems using the train_review.json file. After building the systems, you will use the systems to predict the ratings for a user and business pair. You are required to implement 2 cases:
    Case 1: Item-based CF recommendation system (2.5pts)
    During the predicting process, you will use the system to predict the rating for a given pair of user and business. You must use at most N business neighbors who are the top N most similar to the target business for prediction (you can try various N, e.g., 3 or 5).
    Case 2: User-based CF recommendation system with Min-Hash LSH (2.5pts)
    The predicting process is similar to Case 1.
'''


import sys
import json
import math
from pyspark import SparkContext, SparkConf

    
def func_sort(x):
    x.sort(key=lambda i: -i[1])
    return x


def predict_item(u_b_bs, model_dict, n):
    user, b1 = u_b_bs[0][0], u_b_bs[0][1]
    n_neighbors = []
    for bs in u_b_bs[1]:
        b2 = bs[0]
        star = bs[1]
        pair = tuple(sorted((b1, b2)))
        if pair in model_dict:
            n_neighbors.append([star, model_dict[pair]])
    n_neighbors = func_sort(n_neighbors)[:n]
    numerator = sum([s_s[0] * s_s[1] for s_s in n_neighbors])
    denominator	= sum([abs(s_s[1]) for s_s in n_neighbors])

    pred = 0
    if denominator!=0 and numerator!=0:
        pred = numerator / denominator
    return (user, b1, pred)


def predict_user(b_u_us, model_dict, n, user_avg_dict):
    business, u1 = b_u_us[0][0], b_u_us[0][1]
    n_neighbors = []
    for us in b_u_us[1]:
        u2, star = us[0], us[1]
        pair = tuple(sorted((u1, u2)))
        if pair in model_dict:
            n_neighbors.append([(star, user_avg_dict[u2]), model_dict[pair]])
    n_neighbors = func_sort(n_neighbors)[:n]
    numerator = sum([(s_avg_s[0][0] - s_avg_s[0][1]) * s_avg_s[1] for s_avg_s in n_neighbors])
    denominator = sum([abs(s_avg_s[1]) for s_avg_s in n_neighbors])

    pred = 0
    if denominator!=0 and numerator!=0:
        pred = user_avg_dict[u1] + numerator/denominator
    return (u1, business, pred)


    
def main(params):
    train_file, test_file, model_file = params[1], params[2], params[3]
    output_file, cf_type = params[4], params[5]
    n = 10

    sc = SparkContext.getOrCreate()

    review_train_un = sc.textFile(train_file).map(lambda r: json.loads(r)) \
                                        .map(lambda r: (r['user_id'], r['business_id'], r['stars']))
    review_test_un = sc.textFile(test_file).map(lambda r: json.loads(r)) \
                                        .map(lambda r: (r['user_id'], r['business_id']))
    model_un = sc.textFile(model_file).map(lambda r: json.loads(r))

    if cf_type == 'item_based':
        review_train = review_train_un.map(lambda u_b_s: (u_b_s[0], (u_b_s[1], u_b_s[2]))) \
                                   .groupByKey().map(lambda u_b_s: (u_b_s[0], list(set(u_b_s[1]))))
        review_test = review_test_un.map(lambda u_b: (u_b[0], u_b[1]))
        model = model_un.map(lambda r: (r['b1'], r['b2'], r['sim'])) \
                    .map(lambda b_b_s: ((b_b_s[0], b_b_s[1]), b_b_s[2])).collect()

        model_dict = {}
        for b_b_s in model:
            model_dict[tuple(sorted(b_b_s[0]))] = b_b_s[1]

        pred = review_train.rightOuterJoin(review_test).filter(lambda u_bs_b: u_bs_b[1][0] is not None) \
                           .map(lambda u_bs_b: ((u_bs_b[0], u_bs_b[1][1]), u_bs_b[1][0])) \
                           .filter(lambda u_b_bs: u_b_bs[0][1]!=u_b_bs[1][0]) \
                           .groupByKey().map(lambda u_b_bs: (u_b_bs[0], [b for bs in u_b_bs[1] for b in bs])) \
                           .map(lambda u_b_bs: predict_item(u_b_bs, model_dict, n)).filter(lambda u_b_s: u_b_s[2]!=0).collect()

    if cf_type == 'user_based':
        review_train = review_train_un.map(lambda u_b_s: (u_b_s[1], (u_b_s[0], u_b_s[2]))) \
                                   .groupByKey().map(lambda b_u_s: (b_u_s[0], list(set(b_u_s[1]))))
        review_test = review_test_un.map(lambda u_b: (u_b[1], u_b[0]))
        model = model_un.map(lambda r: (r['u1'], r['u2'], r['sim'])) \
                    .map(lambda u_u_s: ((u_u_s[0], u_u_s[1]), u_u_s[2])).collect()

        model_dict = {}
        for u_u_s in model:
            model_dict[tuple(sorted(u_u_s[0]))] = u_u_s[1]

        user_avg = train_file.split('train_review')[0] + 'user_avg.json'
        user_avg = sc.textFile(user_avg).map(lambda r: json.loads(r)) \
                    .map(lambda u_avg: dict(u_avg)) \
                    .flatMap(lambda u_avg: [(u, avg) for u,avg in u_avg.items()]).collect()

        user_avg_dict = {}
        for u_avg in user_avg:
            user_avg_dict[u_avg[0]] = u_avg[1]

        pred = review_train.rightOuterJoin(review_test).filter(lambda b_us_u: b_us_u[1][0] is not None) \
                        .map(lambda b_us_u: ((b_us_u[0], b_us_u[1][1]), b_us_u[1][0])) \
                        .filter(lambda b_u_us: b_u_us[0][1]!=b_u_us[1][0]) \
                        .groupByKey().map(lambda b_u_us: (b_u_us[0], [u for us in b_u_us[1] for u in us])) \
                        .map(lambda b_u_us: predict_user(b_u_us, model_dict, n, user_avg_dict)) \
                        .filter(lambda u_b_s: u_b_s[2]!=0).collect()

    with open(output_file, 'w+') as f:
        for u_b_s in pred:
            f.write(json.dumps({'user_id': u_b_s[0], 'business_id': u_b_s[1], 'stars': u_b_s[2]}) + '\n')
        
            
if __name__ == '__main__':
    params = sys.argv
    main(params)
    
    