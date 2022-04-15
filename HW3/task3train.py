#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
    Task3: Collaborative Filtering Recommendation System (5pts)

    Task description
    In this task, you will build collaborative filtering (CF) recommendation systems using the train_review.json file. After building the systems, you will use the systems to predict the ratings for a user and business pair. You are required to implement 2 cases:
    Case 1: Item-based CF recommendation system (2.5pts)
    During the training process, you will build a recommendation system by computing the Pearson correlation for the business pairs with at least three co-rated users. 
    Case 2: User-based CF recommendation system with Min-Hash LSH (2.5pts)
    During the training process, you should combine the Min-Hash and LSH algorithms in your user-based CF recommendation system since the number of potential user pairs might be too large to compute. You need to (1) identify user pairs’ similarity using their co-rated businesses without considering their rating scores (similar to Task 1). This process reduces the number of user pairs you need to compare for the final Pearson correlation score. (2) compute the Pearson correlation for the user pair candidates with Jaccard similarity >= 0.01 and at least three co-rated businesses.'''


import sys
import json
import math
import random
from collections import defaultdict
from itertools import combinations
from pyspark import SparkContext, SparkConf



def min_hash(rows, n_hash, p, m):
    hash_dict = defaultdict(list)
    for i in range(len(rows)):
        for j in range(n_hash):
            a, b = random.randint(1, 99999), random.randint(1, 99999)
            hash_dict[i].append(((a*i+b)%p)%m)
    return hash_dict


def LSH(signature, n_hash, n_bands):
    candidates = set()
    n_rows_per_band = n_hash//n_bands
    assert n_hash%n_bands==0, 'Wrong number of bands!'
    for b in range(n_bands):
        buckets = defaultdict(set)
        for sgnt in signature:
            start_row = b * n_rows_per_band
            vector = ()
            for r in range(n_rows_per_band):
                vector += (sgnt[1][start_row+r],)
            vector_hashed = hash(vector)
            buckets[vector_hashed].add(sgnt[0])
        for pairs in buckets.values():
            if len(pairs)>=2:
                for p in combinations(pairs, 2):
                    candidates.add(tuple(sorted(p)))
    return candidates 
    
    
def calc_pearson_sim(user_star1, user_star2):
    user_cor = set(user_star1.keys()) & set(user_star2.keys())
    star1 = [user_star1[u] for u in user_cor]
    star2 = [user_star2[u] for u in user_cor]
    avg1 = sum(star1) / len(star1)
    avg2 = sum(star2) / len(star2)
    star_new1 = [s-avg1 for s in star1]
    star_new2 = [s-avg2 for s in star2]

    numerator = 0
    denominator = math.sqrt(sum([s**2 for s in star_new1])) * math.sqrt(sum([s**2 for s in star_new2]))
    for i in range(len(user_cor)):
        numerator += star_new1[i]*star_new2[i]
    sim = 0
    if denominator>0:
        sim = max(0, numerator/denominator)
    return sim


def calc_jaccard_sim(user_star1, user_star2):
    u1, u2 = set(user_star1.keys()), set(user_star2.keys())
    sim = float(len(u1 & u2) / len(u1 | u2))
    return sim  



def main(params):
    train_file, model_file, cf_type = params[1], params[2], params[3]

    sc = SparkContext.getOrCreate()

    review = sc.textFile(train_file).map(lambda r: json.loads(r)) \
                .map(lambda r: (r['user_id'], r['business_id'], r['stars']))

    business = review.map(lambda u_b_s: u_b_s[1]).distinct().collect()

    if cf_type == 'item_based':
        stars = review.map(lambda u_b_s: (u_b_s[1], (u_b_s[0], u_b_s[2]))) \
                   .groupByKey().map(lambda b_u_s: (b_u_s[0], list(b_u_s[1]))) \
                   .filter(lambda b_u_s: len(b_u_s[1])>=3).collect()
        stars_dict = defaultdict(dict)
        for b_u_s in stars:
            for u_s in b_u_s[1]:
                stars_dict[b_u_s[0]][u_s[0]] = u_s[1]

        with open(model_file, 'w+') as f:
            for p in combinations(business, 2):
                b1, b2 = p[0], p[1]
                u_s1, u_s2 = stars_dict[b1], stars_dict[b2]
                if len(set(u_s1.keys()) & set(u_s2.keys()))>=3:
                    pearson_sim = calc_pearson_sim(u_s1, u_s2)
                    if pearson_sim>0:
                        f.write(json.dumps({'b1': b1, 'b2': b2, 'sim': pearson_sim}) + '\n')

    if cf_type == 'user_based':
        stars = review.map(lambda u_b_s: (u_b_s[0], (u_b_s[1], u_b_s[2]))) \
                   .groupByKey().map(lambda u_b_s: (u_b_s[0], list(u_b_s[1]))) \
                   .filter(lambda u_b_s: len(u_b_s[1])>=3).collect()
        stars_dict = defaultdict(dict)
        for u_b_s in stars:
            for b_s in u_b_s[1]:
                stars_dict[u_b_s[0]][b_s[0]] = b_s[1]

        business_id = {}
        for i,b in enumerate(business):
            business_id[b] = i

        star_index = review.map(lambda u_b_s: (business_id[u_b_s[1]], u_b_s[0]))
        n_hash = 50
        star_hashed = min_hash(business, n_hash, p=9841, m=87319)

        signature = star_index.groupByKey().map(lambda u_bs: (u_bs[0], list(set(u_bs[1])))) \
                            .map(lambda uids_bs: (uids_bs[1], star_hashed[uids_bs[0]])) \
                            .map(lambda bs_uids: ((b, bs_uids[1]) for b in bs_uids[0])).flatMap(lambda b_uids: b_uids) \
                            .groupByKey().map(lambda b_uids: (b_uids[0], [list(uids) for uids in b_uids[1]])) \
                            .map(lambda b_us: (b_us[0], [min(uids) for uids in zip(*b_us[1])])).collect()

        candidates = LSH(signature, n_hash, n_bands=50)
        
        business_users = review.map(lambda u_b: (u_b[1], u_b[0])) \
                                .groupByKey().map(lambda b_u: (b_u[0], set(b_u[1]))).collectAsMap()

        with open(model_file, 'w+') as f:
            for pair in candidates:
                u1, u2 = pair[0], pair[1]
                b_s1, b_s2 = stars_dict[u1], stars_dict[u2]
                if len(set(b_s1.keys()) & set(b_s2.keys()))>=3:
                    pearson_sim = calc_pearson_sim(b_s1, b_s2)
                    if pearson_sim>0 and calc_jaccard_sim(b_s1, b_s2)>=.01:
                        f.write(json.dumps({'u1': u1, 'u2': u2, 'sim': pearson_sim}) + '\n')
                        
        
            
if __name__ == '__main__':
    params = sys.argv
    main(params)
    
    